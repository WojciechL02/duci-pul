#!/usr/bin/env python3
"""
Download ImageNet-1K from Academic Torrents using libtorrent.
"""

import libtorrent as lt
import time
import sys
import os
from pathlib import Path

# Academic Torrents magnet links for ImageNet ILSVRC2012
TORRENTS = {
    "train": "magnet:?xt=urn:btih:a306397ccf9c2ead27155983c254227c0fd938e2&dn=ILSVRC2012_img_train.tar&tr=https://academictorrents.com/announce.php&tr=udp://tracker.coppersurfer.tk:6969&tr=udp://tracker.opentrackr.org:1337/announce",
    "val": "magnet:?xt=urn:btih:5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5&dn=ILSVRC2012_img_val.tar&tr=https://academictorrents.com/announce.php&tr=udp://tracker.coppersurfer.tk:6969&tr=udp://tracker.opentrackr.org:1337/announce",
}

def download_torrent(magnet_link: str, save_path: str, name: str):
    """Download a torrent using libtorrent."""
    
    ses = lt.session()
    ses.listen_on(6881, 6891)
    
    # Add DHT and other settings for better connectivity
    settings = {
        'enable_dht': True,
        'enable_lsd': True,
        'enable_upnp': True,
        'enable_natpmp': True,
    }
    ses.apply_settings(settings)
    
    params = lt.parse_magnet_uri(magnet_link)
    params.save_path = save_path
    
    handle = ses.add_torrent(params)
    print(f"Starting download: {name}")
    print(f"Save path: {save_path}")
    
    # Wait for metadata
    print("Fetching metadata...")
    while not handle.status().has_metadata:
        time.sleep(1)
        s = handle.status()
        print(f"\r  Peers: {s.num_peers}, DHT nodes: {ses.status().dht_nodes}", end="")
    print("\nMetadata received!")
    
    # Download
    print("Downloading...")
    while handle.status().state != lt.torrent_status.seeding:
        s = handle.status()
        progress = s.progress * 100
        download_rate = s.download_rate / 1024 / 1024  # MB/s
        total_size = s.total_wanted / 1024 / 1024 / 1024  # GB
        downloaded = s.total_wanted_done / 1024 / 1024 / 1024  # GB
        
        print(f"\r  {name}: {progress:.1f}% ({downloaded:.1f}/{total_size:.1f} GB) @ {download_rate:.1f} MB/s, peers: {s.num_peers}", end="")
        sys.stdout.flush()
        time.sleep(5)
    
    print(f"\n{name} download complete!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download ImageNet from Academic Torrents')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                        help='Which split to download')
    parser.add_argument('--output', type=str, default='./data/imagenet_download',
                        help='Output directory')
    args = parser.parse_args()
    
    save_path = Path(args.output)
    save_path.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Downloading ImageNet {split} split...")
        print(f"{'='*60}")
        download_torrent(TORRENTS[split], str(save_path), split)
    
    print("\n" + "="*60)
    print("Downloads complete!")
    print(f"Files saved to: {save_path}")
    print("\nNext steps:")
    print("1. Extract the tar files:")
    print(f"   cd {save_path}")
    print("   mkdir -p train val")
    print("   tar -xf ILSVRC2012_img_train.tar -C train/")
    print("   tar -xf ILSVRC2012_img_val.tar -C val/")
    print("2. Extract train class folders:")
    print("   cd train && for f in *.tar; do mkdir -p ${f%.tar} && tar -xf $f -C ${f%.tar}; done")
    print("3. Organize val images into class folders (use valprep.sh script)")


if __name__ == '__main__':
    main()
