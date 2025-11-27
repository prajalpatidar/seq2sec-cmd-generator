#!/usr/bin/env python3
"""Expand dataset by adding variations with different input verbs."""

import json
import os
from copy import deepcopy

# Input variation verbs
INPUT_VERBS = {
    'show': ['display', 'list', 'view', 'get', 'print', 'check', 'find', 'retrieve'],
    'display': ['show', 'list', 'view', 'print', 'check', 'get'],
    'list': ['show', 'display', 'enumerate', 'get', 'find', 'view'],
    'check': ['show', 'display', 'verify', 'view', 'inspect', 'examine'],
    'find': ['search', 'locate', 'get', 'show', 'list', 'retrieve'],
    'get': ['fetch', 'retrieve', 'show', 'display', 'obtain'],
    'create': ['make', 'generate', 'add', 'build', 'establish'],
    'delete': ['remove', 'erase', 'destroy', 'purge', 'clear'],
    'add': ['create', 'insert', 'append', 'include'],
    'remove': ['delete', 'drop', 'erase', 'eliminate'],
}


def generate_variations(input_text, max_variations=3):
    """Generate variations of input text by replacing verbs."""
    variations = []
    words = input_text.lower().split()
    
    for i, word in enumerate(words):
        if word in INPUT_VERBS:
            # Generate variations with different verbs
            for alt_verb in INPUT_VERBS[word][:max_variations]:
                new_words = words.copy()
                new_words[i] = alt_verb
                variations.append(' '.join(new_words))
    
    return variations


def expand_dataset(input_path, output_path, target_size=500):
    """Expand dataset by adding input variations."""
    
    # Load existing dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    original_size = len(data)
    print(f"Original dataset size: {original_size}")
    
    # Create variations
    expanded_data = deepcopy(data)
    added = 0
    
    for item in data:
        if len(expanded_data) >= target_size:
            break
            
        variations = generate_variations(item['input'])
        
        for var_input in variations:
            if len(expanded_data) >= target_size:
                break
            
            # Check if variation already exists
            if not any(d['input'] == var_input for d in expanded_data):
                expanded_data.append({
                    'input': var_input,
                    'output': item['output']
                })
                added += 1
    
    # Additional manual high-quality entries for common operations
    additional_entries = [
        # Network operations
        {"input": "enumerate all network devices", "output": "ip link show"},
        {"input": "view network interface details", "output": "ip addr show"},
        {"input": "inspect network connections", "output": "ss -tuln"},
        {"input": "examine active connections", "output": "netstat -an"},
        {"input": "retrieve routing information", "output": "ip route show"},
        {"input": "fetch dns configuration", "output": "cat /etc/resolv.conf"},
        {"input": "verify network connectivity", "output": "ping -c 4 google.com"},
        {"input": "test connection to host", "output": "ping -c 3"},
        {"input": "trace route to destination", "output": "traceroute"},
        {"input": "scan open ports", "output": "nmap -p-"},
        
        # File operations
        {"input": "enumerate files in directory", "output": "ls -la"},
        {"input": "view file contents", "output": "cat"},
        {"input": "inspect file details", "output": "stat"},
        {"input": "search for files by name", "output": "find . -name"},
        {"input": "locate files in system", "output": "locate"},
        {"input": "retrieve file metadata", "output": "file"},
        {"input": "examine file permissions", "output": "ls -l"},
        {"input": "view hidden files", "output": "ls -la"},
        {"input": "print working directory", "output": "pwd"},
        {"input": "fetch file size", "output": "du -h"},
        
        # System information
        {"input": "view system information", "output": "uname -a"},
        {"input": "inspect cpu information", "output": "lscpu"},
        {"input": "examine memory details", "output": "free -h"},
        {"input": "retrieve disk usage", "output": "df -h"},
        {"input": "fetch uptime information", "output": "uptime"},
        {"input": "view kernel version", "output": "uname -r"},
        {"input": "inspect hardware details", "output": "lshw"},
        {"input": "enumerate pci devices", "output": "lspci"},
        {"input": "view usb devices", "output": "lsusb"},
        {"input": "fetch system logs", "output": "journalctl -xe"},
        
        # Process management
        {"input": "enumerate running processes", "output": "ps aux"},
        {"input": "view process tree", "output": "pstree"},
        {"input": "inspect process details", "output": "ps -ef"},
        {"input": "monitor system processes", "output": "top"},
        {"input": "view resource usage", "output": "htop"},
        {"input": "terminate process by name", "output": "pkill"},
        {"input": "kill process by pid", "output": "kill -9"},
        {"input": "retrieve process id", "output": "pidof"},
        
        # User and permissions
        {"input": "enumerate system users", "output": "cat /etc/passwd"},
        {"input": "view current user", "output": "whoami"},
        {"input": "inspect logged users", "output": "who"},
        {"input": "view user groups", "output": "groups"},
        {"input": "fetch user information", "output": "id"},
        {"input": "examine sudo access", "output": "sudo -l"},
        {"input": "view last login", "output": "last"},
        {"input": "inspect failed logins", "output": "lastb"},
        
        # Disk operations
        {"input": "enumerate disk partitions", "output": "fdisk -l"},
        {"input": "view mounted filesystems", "output": "mount"},
        {"input": "inspect block devices", "output": "lsblk"},
        {"input": "examine disk health", "output": "smartctl -a"},
        {"input": "view inode usage", "output": "df -i"},
        
        # Service management
        {"input": "enumerate active services", "output": "systemctl list-units"},
        {"input": "view service status", "output": "systemctl status"},
        {"input": "inspect running daemons", "output": "systemctl list-units --type=service"},
        {"input": "fetch service logs", "output": "journalctl -u"},
        {"input": "restart system service", "output": "systemctl restart"},
        {"input": "enable service at boot", "output": "systemctl enable"},
        
        # Firewall operations
        {"input": "enumerate firewall rules", "output": "iptables -L -n"},
        {"input": "view nat rules", "output": "iptables -t nat -L"},
        {"input": "inspect firewall status", "output": "ufw status"},
        {"input": "fetch filtering rules", "output": "iptables -L -v"},
        {"input": "view all firewall chains", "output": "iptables -L -n -v"},
        
        # Package management
        {"input": "enumerate installed packages", "output": "dpkg -l"},
        {"input": "view package information", "output": "apt show"},
        {"input": "search for package", "output": "apt search"},
        {"input": "fetch package details", "output": "dpkg -s"},
        {"input": "view package files", "output": "dpkg -L"},
        
        # Archive operations
        {"input": "create tar archive", "output": "tar -czf"},
        {"input": "extract tar file", "output": "tar -xzf"},
        {"input": "view archive contents", "output": "tar -tzf"},
        {"input": "compress with gzip", "output": "gzip"},
        {"input": "decompress gzip file", "output": "gunzip"},
        {"input": "create zip archive", "output": "zip -r"},
        {"input": "extract zip file", "output": "unzip"},
        
        # Text processing
        {"input": "search text in files", "output": "grep -r"},
        {"input": "count lines in file", "output": "wc -l"},
        {"input": "view first lines", "output": "head"},
        {"input": "view last lines", "output": "tail"},
        {"input": "follow log file", "output": "tail -f"},
        {"input": "sort file contents", "output": "sort"},
        {"input": "remove duplicate lines", "output": "uniq"},
        {"input": "compare two files", "output": "diff"},
        
        # Network configuration
        {"input": "configure ip address", "output": "ip addr add"},
        {"input": "bring interface up", "output": "ip link set up"},
        {"input": "bring interface down", "output": "ip link set down"},
        {"input": "add route entry", "output": "ip route add"},
        {"input": "delete route entry", "output": "ip route del"},
        {"input": "flush routing table", "output": "ip route flush"},
        
        # System control
        {"input": "reboot the system", "output": "reboot"},
        {"input": "shutdown system now", "output": "shutdown -h now"},
        {"input": "schedule system shutdown", "output": "shutdown -h +30"},
        {"input": "cancel shutdown", "output": "shutdown -c"},
        {"input": "suspend system", "output": "systemctl suspend"},
        {"input": "hibernate system", "output": "systemctl hibernate"},
    ]
    
    # Add additional entries
    for entry in additional_entries:
        if len(expanded_data) >= target_size:
            break
        if not any(d['input'] == entry['input'] for d in expanded_data):
            expanded_data.append(entry)
            added += 1
    
    # Save expanded dataset
    with open(output_path, 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print(f"Expanded dataset size: {len(expanded_data)}")
    print(f"Added {added} new entries")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "commands-dataset.json")
    output_path = os.path.join(base_dir, "data", "commands-dataset.json")
    
    expand_dataset(input_path, output_path, target_size=500)
