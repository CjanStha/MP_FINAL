import json
import csv
from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
from ...models import Ward


class Command(BaseCommand):
    help = 'Load ward boundaries from CSV file into the database'

    def handle(self, *args, **options):
        csv_path = Path(settings.BASE_DIR).parent / 'data' / 'raw_data' / 'kathmandu_wards_boundary_sorted.csv'
        
        if not csv_path.exists():
            self.stdout.write(self.style.ERROR(f'CSV file not found: {csv_path}'))
            return

        self.stdout.write(self.style.SUCCESS('================================================================================'))
        self.stdout.write(self.style.SUCCESS('LOADING WARD BOUNDARIES'))
        self.stdout.write(self.style.SUCCESS('================================================================================'))

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
                        # Replace escaped double quotes with single quotes temporarily to parse
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
                            self.stdout.write(
                                self.style.SUCCESS(f'  ✓ Loaded {ward.ward_number} ({ward_name})')
                            )
                        else:
                            updated_count += 1
                            self.stdout.write(
                                self.style.WARNING(f'  ↻ Updated {ward.ward_number} ({ward_name})')
                            )
                            
                    except (ValueError, json.JSONDecodeError, KeyError) as e:
                        error_count += 1
                        self.stdout.write(
                            self.style.ERROR(f'  ✗ Error processing row: {e}')
                        )
                        continue

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Failed to read CSV file: {e}'))
            return

        self.stdout.write(self.style.SUCCESS('================================================================================'))
        self.stdout.write(self.style.SUCCESS(f'✅ WARD BOUNDARIES LOADED SUCCESSFULLY'))
        self.stdout.write(self.style.SUCCESS(f'   Loaded: {loaded_count} | Updated: {updated_count} | Errors: {error_count}'))
        self.stdout.write(self.style.SUCCESS('================================================================================'))
