import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv('.env')

url = os.getenv('SUPABASE_URL')
key = (
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    or os.getenv('SUPABASE_KEY')
    or os.getenv('SUPABASE_ANON_KEY')
)

if not url or not key:
    raise SystemExit('Missing SUPABASE_URL or key in .env')

client = create_client(url, key)

buckets_to_ensure = [
    {'id': 'original-images', 'name': 'original-images', 'public': False},
    {'id': 'ai-images', 'name': 'ai-images', 'public': False},
    {'id': 'reports', 'name': 'reports', 'public': False},
]

existing = client.storage.list_buckets() or []
existing_ids = {bucket.get('id') for bucket in existing}

for bucket in buckets_to_ensure:
    if bucket['id'] in existing_ids:
        print(f"Bucket exists: {bucket['id']}")
        continue
    try:
        client.storage.create_bucket(
            bucket['id'],
            bucket['name'],
            {'public': bucket['public']}
        )
        print(f"Created bucket: {bucket['id']}")
    except Exception as exc:
        print(f"Failed bucket {bucket['id']}: {exc}")
