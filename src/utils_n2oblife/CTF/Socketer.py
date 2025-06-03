import socket
import time
import threading
from typing import Optional, Union, Callable


class Socketer:
    """
    A custom socket wrapper class that provides convenient methods for TCP socket operations.
    
    This class simplifies common socket operations like connecting, sending, receiving,
    and handling various data formats commonly used in network programming and CTF challenges.
    """
    
    def __init__(self, ip: str, port: int, delay: float = 0.1, timeout: Optional[float] = None):
        """
        Initialize the Socketer instance and establish connection.
        
        Args:
            ip (str): Target IP address
            port (int): Target port number
            delay (float): Default delay between operations (default: 0.1 seconds)
            timeout (float, optional): Socket timeout in seconds (default: None for blocking)
        """
        self.ip = ip
        self.port = port
        self.delay = delay
        self.timeout = timeout
        self.socket = None
        self.is_connected = False
        self.connect()
    
    def connect(self) -> socket.socket:
        """
        Establish a TCP connection to the target host.
        
        Returns:
            socket.socket: The connected socket object
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.timeout:
                self.socket.settimeout(self.timeout)
            self.socket.connect((self.ip, self.port))
            self.is_connected = True
            return self.socket
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.ip}:{self.port} - {str(e)}")
    
    def disconnect(self) -> None:
        """
        Close the socket connection gracefully.
        """
        if self.socket and self.is_connected:
            try:
                self.socket.close()
            except:
                pass  # Ignore errors during cleanup
            finally:
                self.is_connected = False
    
    def __enter__(self):
        """Context manager entry - returns self for 'with' statements."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup."""
        self.disconnect()
    
    def send(self, message: Union[str, bytes], newline: bool = True) -> None:
        """
        Send a message through the socket.
        
        Args:
            message (Union[str, bytes]): Message to send
            newline (bool): Whether to append newline character (default: True)
            
        Raises:
            ConnectionError: If socket is not connected
        """
        if not self.is_connected:
            raise ConnectionError("Socket is not connected")
        
        if isinstance(message, str):
            message = message.encode()
        
        if newline and not message.endswith(b'\n'):
            message += b'\n'
        
        self.socket.sendall(message)
        time.sleep(self.delay)
    
    def recv(self, size: int = 4096) -> bytes:
        """
        Receive data from the socket.
        
        Args:
            size (int): Maximum number of bytes to receive (default: 4096)
            
        Returns:
            bytes: Received data
            
        Raises:
            ConnectionError: If socket is not connected
        """
        if not self.is_connected:
            raise ConnectionError("Socket is not connected")
        
        return self.socket.recv(size)
    
    def recv_until(self, token: bytes = b"Feedback", max_size: int = 1024*1024) -> bytes:
        """
        Receive data until a specific token is found.
        
        Args:
            token (bytes): Token to search for (default: b"Feedback")
            max_size (int): Maximum total bytes to receive (default: 1MB)
            
        Returns:
            bytes: All received data including the token
            
        Raises:
            ConnectionError: If socket is not connected
            RuntimeError: If max_size is exceeded
        """
        if not self.is_connected:
            raise ConnectionError("Socket is not connected")
        
        data = b""
        while token not in data:
            if len(data) >= max_size:
                raise RuntimeError(f"Exceeded maximum receive size ({max_size} bytes)")
            
            chunk = self.recv(4096)
            if not chunk:
                break
            data += chunk
        
        return data
    
    def recv_line(self, encoding: str = 'utf-8') -> str:
        """
        Receive a single line (until newline character).
        
        Args:
            encoding (str): Text encoding to use (default: 'utf-8')
            
        Returns:
            str: Received line without newline character
        """
        line = self.recv_until(b'\n')
        return line.decode(encoding).rstrip('\n\r')
    
    def recv_all(self, timeout: float = 1.0) -> bytes:
        """
        Receive all available data with a timeout.
        
        Args:
            timeout (float): Timeout in seconds (default: 1.0)
            
        Returns:
            bytes: All received data
        """
        original_timeout = self.socket.gettimeout()
        self.socket.settimeout(timeout)
        
        data = b""
        try:
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                data += chunk
        except socket.timeout:
            pass  # Expected when no more data
        finally:
            self.socket.settimeout(original_timeout)
        
        return data
    
    def send_option(self, option: int) -> None:
        """
        Send a numeric option (commonly used in CTF challenges).
        
        Args:
            option (int): Option number to send
        """
        self.send(str(option))
    
    def capture_packet(self) -> Optional[str]:
        """
        Capture and extract hexadecimal packet data.
        
        This method sends option 1 and extracts hexadecimal data from the response.
        Commonly used for packet capture challenges.
        
        Returns:
            str or None: Hexadecimal packet data if found, None otherwise
        """
        self.send_option(1)
        output = self.recv_until(b"Feedback")  # Fixed: was using self.socket incorrectly
        lines = output.decode().splitlines()
        
        for line in lines:
            stripped = line.strip()
            # Check if line is at least 32 chars and contains only hex characters
            if (len(stripped) >= 32 and 
                all(c in "0123456789abcdef" for c in stripped.lower())):
                return stripped
        
        return None
    
    def send_hex(self, hex_string: str) -> None:
        """
        Send hexadecimal data as bytes.
        
        Args:
            hex_string (str): Hexadecimal string (with or without spaces/colons)
        """
        # Clean hex string (remove spaces, colons, etc.)
        clean_hex = ''.join(c for c in hex_string if c in '0123456789abcdefABCDEF')
        
        # Convert to bytes
        hex_bytes = bytes.fromhex(clean_hex)
        self.send(hex_bytes, newline=False)
    
    def interactive(self) -> None:
        """
        Start an interactive session where user input is sent to the socket
        and socket output is displayed in real-time.
        
        Useful for manual exploration of services.
        Type 'exit' to quit interactive mode.
        """
        print(f"Interactive mode started for {self.ip}:{self.port}")
        print("Type 'exit' to quit")
        
        def recv_thread():
            """Background thread to receive and display data."""
            while self.is_connected:
                try:
                    data = self.recv(4096)
                    if data:
                        print(f"<< {data.decode('utf-8', errors='replace')}", end='')
                except:
                    break
        
        # Start receiving thread
        receiver = threading.Thread(target=recv_thread, daemon=True)
        receiver.start()
        
        try:
            while self.is_connected:
                user_input = input(">> ")
                if user_input.lower() == 'exit':
                    break
                self.send(user_input)
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
        finally:
            self.disconnect()
    
    def probe_service(self) -> dict:
        """
        Probe the service to gather basic information.
        
        Returns:
            dict: Service information including banner, response time, etc.
        """
        start_time = time.time()
        
        # Try to get banner
        banner = self.recv_all(timeout=2.0)
        response_time = time.time() - start_time
        
        info = {
            'host': f"{self.ip}:{self.port}",
            'banner': banner.decode('utf-8', errors='replace') if banner else None,
            'response_time': response_time,
            'connected': self.is_connected
        }
        
        return info
    
    def bruteforce_options(self, start: int = 1, end: int = 10, 
                          success_indicator: bytes = b"success") -> list:
        """
        Bruteforce numeric options to find valid ones.
        
        Args:
            start (int): Starting option number (default: 1)
            end (int): Ending option number (default: 10)
            success_indicator (bytes): Bytes indicating successful option
            
        Returns:
            list: List of valid option numbers
        """
        valid_options = []
        
        for option in range(start, end + 1):
            try:
                self.send_option(option)
                response = self.recv_all(timeout=1.0)
                
                if success_indicator in response:
                    valid_options.append(option)
                    
            except Exception as e:
                print(f"Error testing option {option}: {e}")
                continue
        
        return valid_options
    
    def __repr__(self) -> str:
        """String representation of the Socketer instance."""
        status = "connected" if self.is_connected else "disconnected"
        return f"Socketer({self.ip}:{self.port}, {status})"
    
if __name__ == "__main__":
    # Basic usage
    with Socketer("127.0.0.1", 8080) as sock:
        sock.send("Hello")
        response = sock.recv_line()

        # Interactive exploration
        sock = Socketer("target.com", 1337)
        sock.interactive()  # Manual exploration

        # Bruteforce options
        valid_options = sock.bruteforce_options(1, 20)

        # Hex operations
        sock.send_hex("41424344")  # Sends "ABCD" as bytes