"""
Direct script to load ward boundaries from CSV into Django database
Run from the backend directory: python load_ward_data.py
"""
import os
import django
import json
import csv
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
django.setup()

from api.models import Ward

def load_ward_boundaries():
    """Load ward boundaries from CSV file into database"""
    
    csv_path = Path(__file__).parent.parent / 'data' / 'raw_data' / 'kathmandu_wards_boundary_sorted.csv'
    
    print('=' * 80)
    print('LOADING WARD BOUNDARIES')
    print('=' * 80)
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    print(f"\n[1] Reading CSV from: {csv_path}")
    
    loaded_count = 0
    updated_count = 0
    error_count = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    ward_number = int(row['ward_number'])
                    ward_name = row['ward_name']
                    
                    # Parse the GeoJSON geometry
                    geometry_json = row['geometry_json']
                    # Replace escaped double quotes with actual double quotes
                    geometry_json = geometry_json.replace('""', '"')
                    boundary = json.loads(geometry_json)
                    
                    # Get or create the ward
                    ward, created = Ward.objects.update_or_create(
                        ward_number=ward_number,
                        defaults={
                            'boundary': boundary,
                        }
                    )
                    
                    if created:
                        loaded_count += 1
                        print(f"  ✓ Loaded Ward {ward_number} ({ward_name})")
                    else:
                        updated_count += 1
                        print(f"  ↻ Updated Ward {ward_number} ({ward_name})")
                        
                except (ValueError, json.JSONDecodeError, KeyError) as e:
                    error_count += 1
                    print(f"  ✗ Error processing row: {e}")
                    continue
    
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return
    
    print('\n' + '=' * 80)
    print('✅ WARD BOUNDARIES LOADED SUCCESSFULLY')
    print(f'   Loaded: {loaded_count} | Updated: {updated_count} | Errors: {error_count}')
    print('=' * 80)
    
    # Verify the data
    print("\n[2] Verifying loaded boundaries...")
    total_wards = Ward.objects.count()
    wards_with_boundaries = Ward.objects.exclude(boundary__isnull=True).count()
    print(f"   Total wards: {total_wards}")
    print(f"   Wards with boundaries: {wards_with_boundaries}")
    
    if wards_with_boundaries > 0:
        print("\n✅ Ward boundaries are now available for location validation!")
        print("   Try pinning the Kathmandu Metropolitan Area again.")
    else:
        print("\n❌ No ward boundaries were loaded. Check the CSV file format.")

if __name__ == '__main__':
    load_ward_boundaries()
