import platform
import psutil

def get_system_info():
    # Get OS information
    os_info = platform.uname()
    os_name = os_info.system
    os_version = os_info.version
    os_release = os_info.release

    # Get CPU information
    cpu_info = platform.processor()
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max

    # Get memory information
    mem_info = psutil.virtual_memory()
    total_memory = mem_info.total / (1024 ** 3)  # Convert bytes to GB

    # Get GPU information (if any)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_info = [(gpu.name, gpu.memoryTotal) for gpu in gpus]
    except ImportError:
        gpu_info = "GPUtil not installed"

    # Display the information
    print(f"Operating System: {os_name} {os_version} (Release: {os_release})")
    print(f"Processor: {cpu_info}")
    print(f"Number of Cores: {cpu_count}")
    print(f"Max CPU Frequency: {cpu_freq} MHz")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"GPU Info: {gpu_info}")

if __name__ == "__main__":
    get_system_info()
