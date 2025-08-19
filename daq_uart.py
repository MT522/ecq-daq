"""
UART DAQ Protocol Helper
------------------------
Implements the packet framing described in your PDF:
STX | Command | Flag | SeqNo | SeqInfo | DataLen | Data | CRC | ETX

- STX = 0xAA, ETX = 0x55 (per figures)
- CRC default = XOR-8 across (Command..Data), i.e., header+payload but not STX/CRC/ETX
  (If your firmware uses Dallas/Maxim CRC-8 instead, flip calc_crc to CRC.crc8_dallas_maxim)
- Supports DAQ commands 0x00..0x0F and Printer-Process 0x00..0x04
- Handles stream sequencing bits (Single/First/Middle/Last/Coded)
- Optionally decodes payload with XOR mask if SEQ_CODED bit is set (mask set via commands)

Parsers:
- Fig.3 low-res frames (10 bytes each: fault, addInfo, 8x 1-byte leads)
- Fig.7 high-res frames (18 bytes each: fault, addInfo, 8x int16 leads)
- Fig.6 10-second block header + 18-byte frames (reassembles across multiple packets)

Adjust endianness/signedness if your device differs.
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable, Dict, Any

try:
    import serial  # pyserial
except ImportError as e:
    raise SystemExit("pyserial is required. Install with: pip install pyserial") from e


# -------------------- Framing constants --------------------
STX = 0xAA  # Start-of-packet (per PDF examples)
ETX = 0x55  # End-of-packet


# -------------------- DAQ Command IDs (Table 2 – DAQ) --------------------
DAQ_INIT                  = 0x00  # init_daq
DAQ_TEST                  = 0x01  # test_daq
DAQ_ALIVE                 = 0x02  # alive_daq
DAQ_START_SAMPLING        = 0x03  # start_daq_sampling  (* with sample-rate/bit-count)
DAQ_STOP_SAMPLING         = 0x04  # stop_daq_sampling   (* with sample-rate/bit-count)
DAQ_START_SEND_RDATA      = 0x05  # start_daq_send_rdata (100 sps; Fig.3/7 format)
DAQ_STOP_SEND_RDATA       = 0x06  # stop_daq_send_rdata
DAQ_SET_SAMPLE_RATE       = 0x07  # daq_set_sr (2 bytes)
DAQ_SET_BIT_COUNT         = 0x08  # daq_set_bn (1 byte)
DAQ_ACTIVE_FILTERS        = 0x09  # daq_active_filters (mask payload)
DAQ_DEACTIVE_FILTERS      = 0x0A  # daq_deactive_filters (mask payload)
DAQ_GET_10S_DATA          = 0x0B  # daq_get_10s_data (500 sps; Fig.6)
DAQ_GET_XOR_MASK          = 0x0C  # get_daq_xor_msk
DAQ_SET_XOR_MASK          = 0x0D  # set_daq_xor_msk
DAQ_GET_SERIAL            = 0x0E  # get_daq_serial (Auth., payload optional)
DAQ_GET_INFO              = 0x0F  # get_daq_info (HW/FW info)
# 0x10 - 0xFF reserved


# -------------------- Printer-Process Command IDs (Table 2 – PPROC) --------------------
PPROC_ALIVE               = 0x00  # alive_pproc
PPROC_PRINT               = 0x01  # print_ppros: width(2) + height(2) + gray(1) + pixel-data
PPROC_STOP                = 0x02  # stop_ppros
PPROC_GET_XOR_MASK        = 0x03  # get_pproc_xor_msk
PPROC_SET_XOR_MASK        = 0x04  # set_pproc_xor_msk
# 0x05 - 0xFF reserved


# -------------------- Flag Values (Table 1) --------------------
FLAG_ACK_OK         = 0x00
FLAG_NO_STX         = 0x01
FLAG_NO_ETX         = 0x02
FLAG_BAD_CMD        = 0x03
FLAG_BAD_FLAG       = 0x04
FLAG_BAD_DLEN       = 0x05
FLAG_BAD_DATA       = 0x06
FLAG_CRC_ERR        = 0x07
FLAG_RESEND_STREAM  = 0x08  # *Bad packet-seq-number / packet-seq-info
FLAG_NONE           = 0x80  # Recommended for outbound commands; 0x81-0xFF = command-specific


# -------------------- Packet Sequence Info bits (Fig. 2) --------------------
# Bit positions per Fig. 2:
# #7 Single Packet, #6 First, #5 Middle, #4 Last, #3 Coded?, #2/#1 reserved, #0 reserved
SEQ_SINGLE       = 0x80
SEQ_STREAM_FIRST = 0x40
SEQ_STREAM_MID   = 0x20
SEQ_STREAM_LAST  = 0x10
SEQ_CODED        = 0x08
# bits 0..2 reserved


# -------------------- CRC helpers --------------------
class CRC:
    @staticmethod
    def xor8(data: bytes) -> int:
        """Simple 8-bit XOR of all bytes"""
        c = 0
        for b in data:
            c ^= b
        return c & 0xFF

    @staticmethod
    def crc8_dallas_maxim(data: bytes, init: int = 0x00) -> int:
        """CRC-8 Dallas/Maxim (poly 0x31, reflected)."""
        crc = init
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x01:
                    crc = (crc >> 1) ^ 0x8C  # 0x8C is reversed 0x31
                else:
                    crc >>= 1
        return crc & 0xFF


# Choose the CRC variant your firmware actually uses:
calc_crc = CRC.xor8     # or: calc_crc = CRC.crc8_dallas_maxim


# -------------------- Data classes --------------------
@dataclass
class Packet:
    command: int
    flag: int
    seq_no: int
    seq_info: int
    data: bytes
    crc: int

    def is_single(self) -> bool:
        return bool(self.seq_info & SEQ_SINGLE)

    def is_first(self) -> bool:
        return bool(self.seq_info & SEQ_STREAM_FIRST)

    def is_middle(self) -> bool:
        return bool(self.seq_info & SEQ_STREAM_MID)

    def is_last(self) -> bool:
        return bool(self.seq_info & SEQ_STREAM_LAST)

    def is_coded(self) -> bool:
        return bool(self.seq_info & SEQ_CODED)


# -------------------- Parsing helpers (Fig.3 / Fig.6 / Fig.7) --------------------
def parse_lead_fault(b: int) -> Dict[int, bool]:
    """Returns {lead_index: connected_bool} for 8 leads. 1==connected, 0==disconnected."""
    return {i: bool((b >> i) & 0x01) for i in range(8)}

def parse_additional_info(b: int) -> Dict[str, int]:
    """Only Bit7 (pace) is specified in Fig.5; keep raw for future use."""
    return {"pace": (b >> 7) & 0x01, "raw": b}

def parse_sample_frame_10b(frame: bytes) -> Dict[str, Any]:
    """
    Fig.3 low-res frame (10 bytes total):
     - 1B lead_fault
     - 1B additional_info
     - 8B leads (each 1 byte)
    """
    if len(frame) != 10:
        raise ValueError("Low-res sample frame must be exactly 10 bytes")
    lead_fault = frame[0]
    add_info   = frame[1]
    leads = list(frame[2:10])  # 8 one-byte channels
    return {
        "lead_fault": parse_lead_fault(lead_fault),
        "additional_info": parse_additional_info(add_info),
        "leads": leads
    }

def parse_sample_frame_18b(frame: bytes, signed: bool = True, little_endian: bool = True) -> Dict[str, Any]:
    """
    Fig.7 high-res frame (18 bytes total, also used per-sample in Fig.6 data region):
     - 1B lead_fault
     - 1B additional_info
     - 8 x 2B leads (default: int16 little-endian)
    """
    if len(frame) != 18:
        raise ValueError("High-res sample frame must be exactly 18 bytes")
    lead_fault = frame[0]
    add_info   = frame[1]
    fmt = ('<' if little_endian else '>') + ('8h' if signed else '8H')
    leads = struct.unpack(fmt, frame[2:18])
    return {
        "lead_fault": parse_lead_fault(lead_fault),
        "additional_info": parse_additional_info(add_info),
        "leads": list(leads)
    }


# -------------------- UART Transport + Base Protocol --------------------
class BaseUART:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        self.seq_counter = 0
        self._xor_mask: int = 0x00  # last known XOR mask (for coded payloads)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    # ---- low-level read ----
    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"Timeout while reading {n} bytes (got {len(buf)})")
            buf.extend(chunk)
        return bytes(buf)

    # ---- header packing ----
    @staticmethod
    def _pack_header(command: int, flag: int, seq_no: int, seq_info: int, data: bytes) -> bytes:
        # Header: 1B cmd, 1B flag, 2B seq_no, 1B seq_info, 2B data_len  => 7 bytes
        return struct.pack('<BBHBH', command, flag, seq_no, seq_info, len(data))

    @staticmethod
    def _unpack_header(hdr: bytes) -> Tuple[int, int, int, int, int]:
        command, flag, seq_no, seq_info, data_len = struct.unpack('<BBHBH', hdr)
        return command, flag, seq_no, seq_info, data_len

    # ---- packet build/send/receive ----
    def build_packet(
        self,
        command: int,
        data: bytes = b'',
        flag: int = FLAG_NONE,
        seq_info: int = SEQ_SINGLE,
        seq_no: Optional[int] = None
    ) -> bytes:
        if seq_no is None:
            seq_no = self.seq_counter & 0xFFFF
            self.seq_counter = (self.seq_counter + 1) & 0xFFFF
        hdr = self._pack_header(command, flag, seq_no, seq_info, data)
        crc = calc_crc(hdr + data)  # CRC over Command..Data (header + payload)
        return bytes([STX]) + hdr + data + bytes([crc, ETX])

    def send_command(
        self,
        command: int,
        data: bytes = b'',
        expect_response: bool = True,
        seq_info: int = SEQ_SINGLE
    ) -> Optional["Packet"]:
        pkt = self.build_packet(command, data=data, flag=FLAG_NONE, seq_info=seq_info)
        self.ser.write(pkt)
        self.ser.flush()
        if not expect_response:
            return None
        return self.read_packet()

    def read_packet(self) -> "Packet":
        # Sync to STX
        while True:
            b = self.ser.read(1)
            if not b:
                raise TimeoutError("Timeout waiting for STX")
            if b[0] == STX:
                break

        # Read fixed header (7 bytes)
        hdr = self._read_exact(7)
        command, flag, seq_no, seq_info, dlen = self._unpack_header(hdr)

        # Read payload, CRC, ETX
        data = self._read_exact(dlen)
        crc  = self._read_exact(1)[0]
        etx  = self._read_exact(1)[0]
        if etx != ETX:
            raise ValueError(f"Bad ETX: 0x{etx:02X}")

        # Verify CRC before any decoding
        calc = calc_crc(hdr + data)
        if calc != crc:
            raise ValueError(f"CRC mismatch (calc=0x{calc:02X}, recv=0x{crc:02X})")

        # If coded bit set, decode payload with current XOR mask
        if seq_info & SEQ_CODED:
            data = bytes((b ^ self._xor_mask) for b in data)

        return Packet(command=command, flag=flag, seq_no=seq_no, seq_info=seq_info, data=data, crc=crc)

    # ---- XOR mask management shared by subclasses ----
    @property
    def xor_mask(self) -> int:
        return self._xor_mask

    @xor_mask.setter
    def xor_mask(self, value: int) -> None:
        self._xor_mask = value & 0xFF


# -------------------- DAQ high-level API --------------------
class DAQUart(BaseUART):
    # --- Simple commands (no payload) ---
    def init(self) -> Packet:
        return self.send_command(DAQ_INIT)

    def self_test(self) -> Packet:
        return self.send_command(DAQ_TEST)

    def alive(self) -> Packet:
        return self.send_command(DAQ_ALIVE)

    def start_send_raw(self) -> Packet:
        """Start streaming raw data (100 sps; device may choose Fig.3 or Fig.7 format)."""
        return self.send_command(DAQ_START_SEND_RDATA)

    def stop_send_raw(self) -> Packet:
        return self.send_command(DAQ_STOP_SEND_RDATA)

    def get_10s_data(self) -> List[Packet]:
        """
        Request a 10-second (500 sps) data block (Fig.6).
        Returns the full list of packets (First..Middle..Last).
        """
        first = self.send_command(DAQ_GET_10S_DATA)
        packets = [first]
        if not (first.is_single() or first.is_last()):
            while True:
                pkt = self.read_packet()
                packets.append(pkt)
                if pkt.is_last() or pkt.is_single():
                    break
        return packets

    def get_xor_mask(self) -> Packet:
        resp = self.send_command(DAQ_GET_XOR_MASK)
        # If device returns 1 byte: update local copy
        if resp.data:
            self.xor_mask = resp.data[0]
        return resp

    def set_xor_mask(self, mask: int) -> Packet:
        pkt = self.send_command(DAQ_SET_XOR_MASK, data=bytes([mask & 0xFF]))
        # Typically, device ACKs; also keep local
        self.xor_mask = mask
        return pkt

    def get_serial(self, auth_blob: bytes = b'') -> Packet:
        """Optional auth payload if required."""
        return self.send_command(DAQ_GET_SERIAL, data=auth_blob)

    def get_info(self) -> Packet:
        return self.send_command(DAQ_GET_INFO)

    # --- Parameterized commands ---
    def start_sampling(self, sample_rate_hz: int, bit_count: int) -> Packet:
        """
        Internal start (with sample-rate/bit-count) per footnote.
        Payload: <H (sample_rate)> + <B (bit_count)>
        """
        payload = struct.pack('<HB', sample_rate_hz & 0xFFFF, bit_count & 0xFF)
        return self.send_command(DAQ_START_SAMPLING, data=payload)

    def stop_sampling(self, sample_rate_hz: int, bit_count: int) -> Packet:
        """
        Internal stop (with sample-rate/bit-count) per footnote.
        Payload: <H (sample_rate)> + <B (bit_count)>
        """
        payload = struct.pack('<HB', sample_rate_hz & 0xFFFF, bit_count & 0xFF)
        return self.send_command(DAQ_STOP_SAMPLING, data=payload)

    def set_sample_rate(self, sample_rate_hz: int) -> Packet:
        payload = struct.pack('<H', sample_rate_hz & 0xFFFF)
        return self.send_command(DAQ_SET_SAMPLE_RATE, data=payload)

    def set_bit_count(self, bit_count: int) -> Packet:
        payload = struct.pack('<B', bit_count & 0xFF)
        return self.send_command(DAQ_SET_BIT_COUNT, data=payload)

    def activate_filters(self, mask: int) -> Packet:
        """
        Enable filters by mask (device-specific semantics).
        Using 2 bytes for flexibility; adjust if your FW uses a different size.
        """
        payload = struct.pack('<H', mask & 0xFFFF)
        return self.send_command(DAQ_ACTIVE_FILTERS, data=payload)

    def deactivate_filters(self, mask: int) -> Packet:
        payload = struct.pack('<H', mask & 0xFFFF)
        return self.send_command(DAQ_DEACTIVE_FILTERS, data=payload)

    # --- Streaming parsers ---
    def iter_lowres_stream(self, payload_chunks: Iterable[bytes]) -> Iterable[Dict[str, Any]]:
        """
        Parse concatenated low-res frames (Fig.3). Each frame is 10 bytes.
        Feed this with .data payloads from incoming packets while raw streaming.
        """
        buf = bytearray()
        for chunk in payload_chunks:
            if not chunk:
                continue
            buf.extend(chunk)
            while len(buf) >= 10:
                frame = bytes(buf[:10])
                del buf[:10]
                yield parse_sample_frame_10b(frame)

    def iter_highres_stream(
        self,
        payload_chunks: Iterable[bytes],
        signed: bool = True,
        little_endian: bool = True
    ) -> Iterable[Dict[str, Any]]:
        """
        Parse concatenated high-res frames (Fig.7). Each frame is 18 bytes.
        Feed this with .data payloads from incoming packets while raw streaming.
        """
        buf = bytearray()
        for chunk in payload_chunks:
            if not chunk:
                continue
            buf.extend(chunk)
            while len(buf) >= 18:
                frame = bytes(buf[:18])
                del buf[:18]
                yield parse_sample_frame_18b(frame, signed=signed, little_endian=little_endian)

    def iter_fig6_samples(
        self,
        packets: Iterable[Packet],
        signed: bool = True,
        little_endian: bool = True
    ) -> Iterable[Dict[str, Any]]:
        """
        Parse Fig.6 multi-packet 10s block.
        Expected data layout across concatenated packets:
          SampleCount(2) | SampleRate(2) | BitCount(1) | AddInfo(1) | [18B sample]*N
        Yields one dict per 18-byte sample frame (same format as Fig.7).
        """
        buf = bytearray()
        total_expected: Optional[int] = None

        for pkt in packets:
            if pkt.data:
                buf.extend(pkt.data)

            # Parse header once (first 6 bytes)
            if total_expected is None and len(buf) >= 6:
                total_expected = int.from_bytes(buf[0:2], 'little')
                _sr = int.from_bytes(buf[2:4], 'little')
                _bit_count = buf[4]
                _block_add_info = buf[5]  # global block add-info; per-frames still have their own add-info
                del buf[:6]

            # Emit frames whenever we have a full 18-byte chunk
            while len(buf) >= 18:
                frame = bytes(buf[:18])
                del buf[:18]
                yield parse_sample_frame_18b(frame, signed=signed, little_endian=little_endian)


# -------------------- Printer-Process API --------------------
class PrinterUART(BaseUART):
    def alive(self) -> Packet:
        return self.send_command(PPROC_ALIVE)

    def stop(self) -> Packet:
        return self.send_command(PPROC_STOP)

    def get_xor_mask(self) -> Packet:
        resp = self.send_command(PPROC_GET_XOR_MASK)
        if resp.data:
            self.xor_mask = resp.data[0]
        return resp

    def set_xor_mask(self, mask: int) -> Packet:
        pkt = self.send_command(PPROC_SET_XOR_MASK, data=bytes([mask & 0xFF]))
        self.xor_mask = mask
        return pkt

    def print_rendered(self, width: int, height: int, gray_levels: int, pixel_data: bytes) -> Packet:
        """
        Send rendered image to printer-process.
        Payload format (per PDF):
           width (2B) + height (2B) + gray-scale-level (1B) + pixel-data
        """
        payload = struct.pack('<HHB', width & 0xFFFF, height & 0xFFFF, gray_levels & 0xFF) + pixel_data
        return self.send_command(PPROC_PRINT, data=payload)


# -------------------- Example usage --------------------
if __name__ == "__main__":
    # Edit to match your system:
    DAQ_PORT = "COM4"      # e.g. "/dev/ttyUSB0", "/dev/ttyACM0", "/dev/tty.usbmodemXXXX"
    BAUD     = 115200

    daq = DAQUart(DAQ_PORT, BAUD)
    try:
        print("-> DAQ Init")
        r = daq.init()
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")

        print("-> DAQ Self-test")
        r = daq.self_test()
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")

        print("-> DAQ ALIVE")
        r = daq.alive()
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")

        # Set parameters (optional)
        print("-> set sample rate = 500 Hz, bit-count = 16")
        r = daq.set_sample_rate(500)
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")
        r = daq.set_bit_count(16)
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")

        # Start raw stream briefly and collect a few payload chunks
        print("-> start raw stream")
        r = daq.start_send_raw()
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")

        t_end = time.time() + 2.0
        chunks: List[bytes] = []
        while time.time() < t_end:
            try:
                pkt = daq.read_packet()
            except TimeoutError:
                continue
            if pkt.flag == FLAG_RESEND_STREAM:
                print("!! device requested resend/sequence recovery")
            if pkt.data:
                chunks.append(pkt.data)

        print(f"Collected {len(chunks)} raw payload chunks")
        print("-> stop raw stream")
        daq.stop_send_raw()
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")

        # Try parsing as high-res frames (18B). If your device streams low-res, use iter_lowres_stream.
        count = 0
        for frame in daq.iter_highres_stream(chunks):
            if count < 5:
                print(f"  HR frame {count}: L1={frame['leads'][0]} pace={frame['additional_info']['pace']}")
            count += 1
        print(f"Parsed {count} high-res frames (if zero, your stream may be low-res 10B frames).")

        # Request a 10s block (Fig.6) and parse a few frames
        print("-> request 10s block")
        block = daq.get_10s_data()
        print(f"<- resp: cmd=0x{r.command:02X} flag=0x{r.flag:02X} seq={r.seq_no} len={len(r.data)}")
        i = 0
        for sample in daq.iter_fig6_samples(block):
            if i < 5:
                print(f"  10s sample {i}: L1={sample['leads'][0]} pace={sample['additional_info']['pace']}")
            i += 1
        print(f"Parsed {i} Fig.6 frames from 10s block")

    finally:
        daq.close()
