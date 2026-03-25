from rest_framework import serializers
from .models import Cafe, Ward, UserProfile, Amenity, AnalysisHistory
from .location_validation import is_within_kathmandu_metropolitan_city
import math


# ═══════════════════════════════════════════════════════════════════
# CafeSerializer - Enhanced with Rating & Review Analysis
# Converts a Cafe database object → JSON for the frontend
# ═══════════════════════════════════════════════════════════════════
class CafeSerializer(serializers.ModelSerializer):

    # Extra field not in the model — calculated from rating × log(reviews)
    score = serializers.SerializerMethodField()
    
    # Enhanced rating fields
    star_rating = serializers.SerializerMethodField()
    rating_category = serializers.SerializerMethodField()
    rating_text = serializers.SerializerMethodField()
    review_summary = serializers.SerializerMethodField()

    class Meta:
        model  = Cafe
        fields = [
            'id',
            'place_id',
            'name',
            'cafe_type',
            'latitude',
            'longitude',
            'rating',
            'review_count',
            'is_open',
            'score',
            'star_rating',
            'rating_category',
            'rating_text',
            'review_summary',
        ]

    def get_score(self, obj):
        """Weighted score used to rank Top 5 cafes.
        Formula: rating × log(review_count + 1)
        """
        if obj.rating is None:
            return 0
        return round(obj.rating * math.log(obj.review_count + 1), 2)

    def get_star_rating(self, obj):
        """Convert numeric rating to star emoji representation"""
        if obj.rating is None:
            return '⭐ N/A'
        
        full_stars = int(obj.rating)
        remainder = obj.rating - full_stars
        
        stars = '⭐' * full_stars
        if remainder >= 0.5:
            stars += '✨'  # Half star
        
        return f"{stars} {obj.rating}"

    def get_rating_category(self, obj):
        """Categorize rating quality"""
        if obj.rating is None:
            return 'Not Rated'
        elif obj.rating >= 4.5:
            return 'Excellent'
        elif obj.rating >= 4.0:
            return 'Very Good'
        elif obj.rating >= 3.5:
            return 'Good'
        elif obj.rating >= 3.0:
            return 'Average'
        else:
            return 'Below Average'

    def get_rating_text(self, obj):
        """Provide human-friendly rating text"""
        category = self.get_rating_category(obj)
        if obj.review_count and obj.review_count > 0:
            return f"{category} - {obj.rating}/5 ({obj.review_count} reviews)"
        else:
            return f"{category} - {obj.rating}/5 (No reviews yet)"

    def get_review_summary(self, obj):
        """Provide review engagement analysis"""
        if obj.review_count is None or obj.review_count == 0:
            return "No customer reviews yet"
        elif obj.review_count < 10:
            return f"Early stage - {obj.review_count} review(s)"
        elif obj.review_count < 50:
            return f"Growing popularity - {obj.review_count} reviews"
        elif obj.review_count < 100:
            return f"Popular - {obj.review_count} customer reviews"
        else:
            return f"Well-established - {obj.review_count} customer reviews"


# ═══════════════════════════════════════════════════════════════════
# AmenitySerializer
# Converts Amenity objects (schools, hospitals, bus stops) to JSON
# ═══════════════════════════════════════════════════════════════════
class AmenitySerializer(serializers.ModelSerializer):

    class Meta:
        model  = Amenity
        fields = [
            'id',
            'osm_id',
            'amenity_type',
            'name',
            'latitude',
            'longitude',
        ]


# ═══════════════════════════════════════════════════════════════════
# WardSerializer
# Converts Ward demographic data to JSON
# ═══════════════════════════════════════════════════════════════════
class WardSerializer(serializers.ModelSerializer):
    
    population_formatted = serializers.SerializerMethodField()
    density_category = serializers.SerializerMethodField()
    households_per_capita = serializers.SerializerMethodField()

    class Meta:
        model = Ward
        fields = [
            'ward_number',
            'population',
            'population_formatted',
            'households',
            'area_sqkm',
            'population_density',
            'density_category',
            'households_per_capita',
        ]

    def get_population_formatted(self, obj):
        """Format population with thousands separator"""
        if obj.population:
            return f"{obj.population:,}"
        return "0"

    def get_density_category(self, obj):
        """Categorize population density"""
        density = obj.population_density
        if density < 8000:
            return 'Low Density'
        elif density < 12000:
            return 'Moderate Density'
        else:
            return 'High Density'

    def get_households_per_capita(self, obj):
        """Calculate average household size"""
        if obj.households and obj.population and obj.households > 0:
            return round(obj.population / obj.households, 2)
        return 0


# ═══════════════════════════════════════════════════════════════════
# DemographicInfoSerializer
# Aggregated demographic analysis for suitability analysis
# ═══════════════════════════════════════════════════════════════════
class DemographicInfoSerializer(serializers.Serializer):
    """Serializer for demographic information (not a model)"""
    
    ward_number = serializers.IntegerField()
    population = serializers.IntegerField()
    population_density = serializers.FloatField()
    households = serializers.IntegerField()
    average_household_size = serializers.FloatField()
    area_sqkm = serializers.FloatField()
    density_category = serializers.CharField()
    population_category = serializers.CharField()
    market_potential = serializers.CharField()


# ═══════════════════════════════════════════════════════════════════
# SuitabilityRequestSerializer
# Validates the incoming POST data when user pins a location
# ═══════════════════════════════════════════════════════════════════
class SuitabilityRequestSerializer(serializers.Serializer):
    # Frontend sends: { "lat": 27.7172, "lng": 85.3240, "cafe_type": "bakery", "radius": 500 }

    lat       = serializers.FloatField(
        min_value=27.6, max_value=27.8,  # Kathmandu latitude range
        error_messages={'min_value': 'Location must be within Kathmandu.'}
    )
    lng       = serializers.FloatField(
        min_value=85.2, max_value=85.5,  # Kathmandu longitude range
    )
    cafe_type = serializers.ChoiceField(
        choices=['coffee_shop', 'bakery', 'dessert_shop', 'restaurant']
    )
    radius    = serializers.IntegerField(
        min_value=100, max_value=2000, default=500,  # 100m to 2km radius
        required=False
    )

    def validate(self, attrs):
        attrs = super().validate(attrs)
        if not is_within_kathmandu_metropolitan_city(attrs['lat'], attrs['lng']):
            raise serializers.ValidationError({
                'lat': 'Location pinning is allowed only inside Kathmandu Metropolitan City.'
            })
        return attrs


# ═══════════════════════════════════════════════════════════════════
# UserProfileSerializer
# ═══════════════════════════════════════════════════════════════════
class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UserProfile
        fields = ['id', 'username', 'email', 'date_joined', 'is_active', 'first_name', 'last_name']
        read_only_fields = ['date_joined', 'is_active']  # can't be set by the client


class AnalysisHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisHistory
        fields = [
            'id',
            'latitude',
            'longitude',
            'cafe_type',
            'radius',
            'suitability_score',
            'suitability_level',
            'created_at',
        ]
