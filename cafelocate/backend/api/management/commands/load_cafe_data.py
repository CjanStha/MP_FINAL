"""
Management command to load cafe and census data from CSV files.

Usage:
    python manage.py load_cafe_data --cafes path/to/kathmandu_cafes.csv --census path/to/kathmandu_census.csv
"""

import os
import json
import hashlib
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from api.models import Cafe, Ward


class Command(BaseCommand):
    help = 'Load cafe and census data from CSV files into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--cafes',
            type=str,
            help='Path to kathmandu_cafes.csv file',
            default=None,
        )
        parser.add_argument(
            '--census',
            type=str,
            help='Path to kathmandu_census.csv file',
            default=None,
        )
        parser.add_argument(
            '--clear-cafes',
            action='store_true',
            help='Clear all existing cafes before loading',
        )
        parser.add_argument(
            '--clear-wards',
            action='store_true',
            help='Clear all existing wards before loading',
        )

    def generate_place_id(self, name, lat, lng):
        """Generate a unique place_id from name, lat, lng"""
        hash_input = f"{name}:{lat:.6f}:{lng:.6f}"
        hash_obj = hashlib.md5(hash_input.encode())
        return f"gid_{hash_obj.hexdigest()[:16]}"

    def infer_cafe_type(self, name):
        """Infer cafe type from the name"""
        name_lower = name.lower()
        
        if any(word in name_lower for word in ['bakery', 'bread', 'pastry']):
            return 'bakery'
        elif any(word in name_lower for word in ['dessert', 'ice cream', 'cake', 'sweet']):
            return 'dessert_shop'
        elif any(word in name_lower for word in ['restaurant', 'diner', 'eatery', 'cuisine']):
            return 'restaurant'
        else:
            # Default to coffee_shop
            return 'coffee_shop'

    def load_cafes(self, csv_path, options):
        """Load cafes from CSV file"""
        if not os.path.exists(csv_path):
            raise CommandError(f'Cafe CSV file not found: {csv_path}')

        self.stdout.write(f'Reading cafe data from {csv_path}...')
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise CommandError(f'Error reading CSV file: {e}')

        # Validate required columns
        required_cols = ['name', 'rating', 'review_count', 'lat', 'lng']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise CommandError(f'Missing required columns: {missing_cols}')

        # Clear existing cafes if requested
        if options['clear_cafes']:
            count = Cafe.objects.count()
            Cafe.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'Cleared {count} existing cafes'))

        created_count = 0
        updated_count = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            try:
                place_id = self.generate_place_id(row['name'], row['lat'], row['lng'])
                cafe_type = self.infer_cafe_type(row['name'])
                
                # Handle NaN values
                rating = float(row['rating']) if pd.notna(row['rating']) else None
                review_count = int(row['review_count']) if pd.notna(row['review_count']) else 0
                
                # Create GeoJSON location
                location = {
                    'type': 'Point',
                    'coordinates': [float(row['lng']), float(row['lat'])]
                }

                cafe, created = Cafe.objects.update_or_create(
                    place_id=place_id,
                    defaults={
                        'name': row['name'],
                        'cafe_type': cafe_type,
                        'latitude': float(row['lat']),
                        'longitude': float(row['lng']),
                        'location': location,
                        'rating': rating,
                        'review_count': review_count,
                        'is_open': True,
                        'collected_at': timezone.now(),
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    updated_count += 1

                if (idx + 1) % 500 == 0:
                    self.stdout.write(f'Processed {idx + 1}/{len(df)} cafes...')

            except Exception as e:
                skipped_count += 1
                if skipped_count <= 5:  # Print first 5 errors
                    self.stdout.write(
                        self.style.WARNING(f'Skipped cafe {idx}: {str(e)[:100]}')
                    )
                continue

        self.stdout.write(
            self.style.SUCCESS(
                f'\n✓ Cafes loaded successfully!'
                f'\n  Created: {created_count}'
                f'\n  Updated: {updated_count}'
                f'\n  Skipped: {skipped_count}'
                f'\n  Total in database: {Cafe.objects.count()}'
            )
        )

    def load_census(self, csv_path, options):
        """Load census data from CSV file"""
        if not os.path.exists(csv_path):
            raise CommandError(f'Census CSV file not found: {csv_path}')

        self.stdout.write(f'\nReading census data from {csv_path}...')
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise CommandError(f'Error reading CSV file: {e}')

        # Validate required columns
        required_cols = ['ward_no', 'population', 'households', 'area_sqkm', 'population_density']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise CommandError(f'Missing required columns: {missing_cols}')

        # Clear existing wards if requested
        if options['clear_wards']:
            count = Ward.objects.count()
            Ward.objects.all().delete()
            self.stdout.write(self.style.SUCCESS(f'Cleared {count} existing wards'))

        created_count = 0
        updated_count = 0
        skipped_count = 0

        for idx, row in df.iterrows():
            try:
                ward, created = Ward.objects.update_or_create(
                    ward_number=int(row['ward_no']),
                    defaults={
                        'population': int(row['population']),
                        'households': int(row['households']),
                        'area_sqkm': float(row['area_sqkm']),
                        'population_density': float(row['population_density']),
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    updated_count += 1

            except Exception as e:
                skipped_count += 1
                self.stdout.write(
                    self.style.WARNING(f'Skipped ward {idx}: {str(e)[:100]}')
                )
                continue

        self.stdout.write(
            self.style.SUCCESS(
                f'\n✓ Census data loaded successfully!'
                f'\n  Created: {created_count}'
                f'\n  Updated: {updated_count}'
                f'\n  Skipped: {skipped_count}'
                f'\n  Total wards in database: {Ward.objects.count()}'
            )
        )

    def handle(self, *args, **options):
        cafe_csv = options.get('cafes')
        census_csv = options.get('census')

        if not cafe_csv and not census_csv:
            raise CommandError(
                'Please provide at least one CSV file using --cafes or --census\n'
                'Example: python manage.py load_cafe_data --cafes path/to/cafes.csv --census path/to/census.csv'
            )

        if cafe_csv:
            self.load_cafes(cafe_csv, options)

        if census_csv:
            self.load_census(census_csv, options)

        self.stdout.write('\n' + self.style.SUCCESS('✓ All data loaded successfully!'))
