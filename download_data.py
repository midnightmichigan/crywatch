"""Download the baby-cry dataset via HF resolve URLs (urllib, threaded)."""
import os
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE, 'data', 'baby-cry')
URL = ('https://huggingface.co/datasets/mahmudulhasan01/baby_crying_sound'
       '/resolve/main/{}')


def fetch(rel):
    dest = os.path.join(DEST, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        with open(dest, 'rb') as fh:
            if fh.read(4) == b'RIFF':
                return 'skip'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        url = URL.format(urllib.parse.quote(rel))
        req = urllib.request.Request(url,
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if data[:4] != b'RIFF':
            return 'bad:' + rel
        with open(dest, 'wb') as fh:
            fh.write(data)
        return 'ok'
    except Exception as e:
        return f'err:{rel}:{e}'


def main():
    import sys
    with open(os.path.join(BASE, 'data', 'filelist.txt')) as f:
        rels = [l.strip() for l in f if l.strip()]
    if len(sys.argv) == 3:
        rels = rels[int(sys.argv[1]):int(sys.argv[2])]
    from collections import Counter
    counts = Counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        for i, res in enumerate(ex.map(fetch, rels)):
            counts[res.split(':')[0]] += 1
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(rels)}', dict(counts), flush=True)
    print('final:', dict(counts), flush=True)


if __name__ == '__main__':
    main()
