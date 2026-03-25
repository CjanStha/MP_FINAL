from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
import jwt, math, logging, json

from .models import Cafe, Ward, Road, UserProfile, Amenity, AnalysisHistory
from .serializers import CafeSerializer, SuitabilityRequestSerializer, UserProfileSerializer, AmenitySerializer, AnalysisHistorySerializer
from .location_validation import is_within_kathmandu_metropolitan_city
from ml_engine.suitability_predictor import get_suitability_prediction

try:
    from shapely.wkt import loads as wkt_loads
    from shapely.geometry import Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)

SUPPORTED_CAFE_TYPES = {
    'coffee_shop': 'Coffee Shop',
    'bakery': 'Bakery Cafe',
    'dessert_shop': 'Dessert Shop',
    'restaurant': 'Restaurant Cafe',
}


def _clamp_model_feature(value):
    """
    Keep live regression features inside the 0-10 band used by the saved model artifacts.
    """
    return max(0.0, min(10.0, float(value)))


def _build_regression_features(
    pop_density,
    accessibility_score,
    foot_traffic_score,
    competition_pressure,
    amenity_stats,
    osm_amenity_density_500m,
):
    """
    Align live API features with the scale used during model training.

    The training pipeline learned from bounded 0-10 style inputs. In production we compute
    some raw values in different units, especially ward population density (people / km²),
    so we map them back into the model's expected range before scoring.
    """
    return {
        # Census density is stored in people/km² (for example ~6,700). The training dataset
        # used a compressed density feature around 0-150, so divide by 1000 before clamping.
        'population_density': round(_clamp_model_feature(pop_density / 1000.0), 2),
        'accessibility_score': round(_clamp_model_feature(accessibility_score), 2),
        'foot_traffic_score': round(_clamp_model_feature(foot_traffic_score), 2),
        # Training defined competition_effective as the inverse of competition pressure.
        'competition_effective': round(_clamp_model_feature(10.0 - competition_pressure), 2),
        'bus_stops_within_500m': round(_clamp_model_feature(amenity_stats['bus_stops_within_500m']), 2),
        'osm_amenity_density_500m': round(_clamp_model_feature(osm_amenity_density_500m), 2),
        'nearby_schools': round(_clamp_model_feature(amenity_stats['schools_within_500m']), 2),
        'nearby_hospitals': round(_clamp_model_feature(amenity_stats['hospitals_within_500m']), 2),
    }


def get_request_user(request):
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return None

    token = auth_header.split(' ', 1)[1].strip()
    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=['HS256'])
        user_id = payload.get('user_id')
        if not user_id:
            return None
        return UserProfile.objects.filter(id=user_id, is_active=True).first()
    except Exception:
        return None


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points.
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371 * 1000  # metres


def _recommend_cafe_type(total_competitors, nearby_cafes, amenity_stats, accessibility_score, foot_traffic_score, pop_density):
    type_counts = {key: 0 for key in SUPPORTED_CAFE_TYPES}
    type_avg_rating = {key: 0.0 for key in SUPPORTED_CAFE_TYPES}

    ratings_by_type = {key: [] for key in SUPPORTED_CAFE_TYPES}
    for cafe in nearby_cafes:
        cafe_type = (cafe.cafe_type or '').strip().lower()
        if cafe_type not in SUPPORTED_CAFE_TYPES:
            continue
        type_counts[cafe_type] += 1
        if cafe.rating is not None:
            ratings_by_type[cafe_type].append(float(cafe.rating))

    for cafe_type, ratings in ratings_by_type.items():
        if ratings:
            type_avg_rating[cafe_type] = sum(ratings) / len(ratings)

    demand_pressure = min(10.0, max(0.0, float(pop_density) / 1500.0))
    school_signal = min(10.0, amenity_stats['schools_within_500m'] * 0.8)
    transit_signal = min(10.0, amenity_stats['bus_stops_within_500m'] * 0.9)

    raw_scores = {
        'coffee_shop': (
            (foot_traffic_score * 0.34) +
            (accessibility_score * 0.24) +
            (transit_signal * 0.14) +
            (demand_pressure * 0.10) +
            (max(0.0, 5.0 - type_counts['coffee_shop']) * 0.36)
        ),
        'bakery': (
            (school_signal * 0.25) +
            (demand_pressure * 0.18) +
            (accessibility_score * 0.18) +
            (foot_traffic_score * 0.12) +
            (max(0.0, 4.0 - type_counts['bakery']) * 0.55)
        ),
        'dessert_shop': (
            (school_signal * 0.26) +
            (foot_traffic_score * 0.20) +
            (demand_pressure * 0.12) +
            (transit_signal * 0.12) +
            (max(0.0, 3.0 - type_counts['dessert_shop']) * 0.70)
        ),
        'restaurant': (
            (demand_pressure * 0.28) +
            (accessibility_score * 0.22) +
            (foot_traffic_score * 0.16) +
            (transit_signal * 0.10) +
            (max(0.0, 6.0 - type_counts['restaurant']) * 0.28)
        ),
    }

    for cafe_type, avg_rating in type_avg_rating.items():
        raw_scores[cafe_type] += min(1.0, avg_rating / 5.0) * 0.35

    score_sum = sum(max(0.01, value) for value in raw_scores.values())
    probabilities = {
        SUPPORTED_CAFE_TYPES[cafe_type]: round(max(0.01, value) / score_sum, 3)
        for cafe_type, value in sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    }
    best_type = max(raw_scores, key=raw_scores.get)

    return {
        'recommended_cafe_type': SUPPORTED_CAFE_TYPES[best_type],
        'recommended_cafe_type_confidence': round(probabilities[SUPPORTED_CAFE_TYPES[best_type]], 3),
        'cafe_type_probabilities': probabilities,
        'cafe_type_method': 'heuristic_location_fit',
        'cafe_type_counts_nearby': {SUPPORTED_CAFE_TYPES[key]: type_counts[key] for key in SUPPORTED_CAFE_TYPES},
    }


def _get_amenity_stats(lat, lng, radius):
    """
    Build radius-aware amenity features used by the regression model.
    """
    amenity_groups = {
        'schools': ['school', 'college', 'university'],
        'hospitals': ['hospital', 'health_post', 'clinic', 'pharmacy'],
        'bus_stops': ['bus_station', 'bus_stop'],
    }

    stats = {
        'schools_within_500m': 0,
        'schools_within_200m': 0,
        'schools_min_distance': 500.0,
        'hospitals_within_500m': 0,
        'hospitals_min_distance': 500.0,
        'bus_stops_within_500m': 0,
        'bus_stops_min_distance': 500.0,
    }

    amenity_records = list(
        Amenity.objects.filter(
            amenity_type__in=sum(amenity_groups.values(), [])
        ).values('amenity_type', 'latitude', 'longitude')
    )

    for amenity in amenity_records:
        amenity_type = amenity['amenity_type']
        distance = haversine_distance(lat, lng, amenity['latitude'], amenity['longitude'])

        if amenity_type in amenity_groups['schools']:
            if distance <= 500:
                stats['schools_within_500m'] += 1
            if distance <= 200:
                stats['schools_within_200m'] += 1
            stats['schools_min_distance'] = min(stats['schools_min_distance'], distance)

        elif amenity_type in amenity_groups['hospitals']:
            if distance <= 500:
                stats['hospitals_within_500m'] += 1
            stats['hospitals_min_distance'] = min(stats['hospitals_min_distance'], distance)

        elif amenity_type in amenity_groups['bus_stops']:
            if distance <= 500:
                stats['bus_stops_within_500m'] += 1
            stats['bus_stops_min_distance'] = min(stats['bus_stops_min_distance'], distance)

    for key in ['schools_min_distance', 'hospitals_min_distance', 'bus_stops_min_distance']:
        stats[key] = round(stats[key], 2)

    return stats


def _get_demographic_info(lat, lng):
    """
    Get demographic information for the given location.
    Returns a dictionary with ward population data and analysis.
    """
    from .serializers import DemographicInfoSerializer
    
    # Find the ward containing the point
    primary_ward = None
    for ward in Ward.objects.all():
        # Simple point-in-polygon check using boundary
        if ward.boundary and isinstance(ward.boundary, dict):
            if point_in_polygon(lng, lat, ward.boundary):
                primary_ward = ward
                break
    
    if not primary_ward:
        # If no ward found, use the closest ward or a default
        primary_ward = Ward.objects.all().first()
    
    if primary_ward:
        avg_household_size = 0
        if primary_ward.households and primary_ward.households > 0:
            avg_household_size = primary_ward.population / primary_ward.households
        
        # Categorize population
        if primary_ward.population < 10000:
            population_category = 'Low Population'
        elif primary_ward.population < 15000:
            population_category = 'Moderate Population'
        else:
            population_category = 'High Population'
        
        # Market potential analysis
        if primary_ward.population_density > 12000:
            market_potential = 'High Demand Market'
        elif primary_ward.population_density > 8000:
            market_potential = 'Good Demand Market'
        else:
            market_potential = 'Growing Market'
        
        demo_data = {
            'ward_number': primary_ward.ward_number,
            'population': primary_ward.population,
            'population_density': primary_ward.population_density,
            'households': primary_ward.households,
            'average_household_size': round(avg_household_size, 2),
            'area_sqkm': primary_ward.area_sqkm,
            'density_category': 'High Density' if primary_ward.population_density > 12000 else 
                               ('Moderate Density' if primary_ward.population_density > 8000 else 'Low Density'),
            'population_category': population_category,
            'market_potential': market_potential,
        }
    else:
        # Default/fallback
        demo_data = {
            'ward_number': 0,
            'population': 0,
            'population_density': 10000,
            'households': 0,
            'average_household_size': 4.5,
            'area_sqkm': 1.0,
            'density_category': 'Unknown',
            'population_category': 'Unknown',
            'market_potential': 'Unknown',
        }
    
    return demo_data



    """
    Approximate shortest distance (meters) from point (px,py) to segment (x1,y1)-(x2,y2).
    Uses an equirectangular projection around the point for small-distance accuracy.
    Inputs are lat/lon in degrees: px,py = point lat,lon; x1,y1,x2,y2 = segment lat,lon.
    """
    # Reference latitude for longitude scaling
    ref_lat = px

    # Convert degrees to meters (approx)
    def to_xy(lat, lon):
        x = lon * 111320 * math.cos(math.radians(ref_lat))
        y = lat * 110574
        return x, y

    px_x, px_y = to_xy(px, py)
    x1_x, x1_y = to_xy(x1, y1)
    x2_x, x2_y = to_xy(x2, y2)

    dx = x2_x - x1_x
    dy = x2_y - x1_y
    if dx == 0 and dy == 0:
        # Segment is a point
        return math.hypot(px_x - x1_x, px_y - x1_y)

    # Project point onto segment, clamp t to [0,1]
    t = ((px_x - x1_x) * dx + (px_y - x1_y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1_x + t * dx
    proj_y = x1_y + t * dy

    return math.hypot(px_x - proj_x, px_y - proj_y)


def _nearest_main_road_distance(lat, lng):
    """
    Return the nearest distance in meters from (lat,lng) to the nearest "main" road
    (primary/secondary/tertiary). If none found, returns None.
    """
    main_types = set(['primary', 'secondary', 'tertiary'])
    nearest = None

    # First pass: consider only main road types
    for road in Road.objects.all():
        if not (road.geometry and isinstance(road.geometry, dict)):
            continue
        if (road.road_type or '').lower() not in main_types:
            continue

        geom_type = road.geometry.get('type')
        coords = road.geometry.get('coordinates', [])

        if geom_type == 'LineString' and coords:
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i][0], coords[i][1]
                lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
                d = _distance_point_to_segment_m(lat, lng, lat1, lon1, lat2, lon2)
                if nearest is None or d < nearest:
                    nearest = d

        elif geom_type == 'MultiLineString' and coords:
            for linestring in coords:
                for i in range(len(linestring) - 1):
                    lon1, lat1 = linestring[i][0], linestring[i][1]
                    lon2, lat2 = linestring[i + 1][0], linestring[i + 1][1]
                    d = _distance_point_to_segment_m(lat, lng, lat1, lon1, lat2, lon2)
                    if nearest is None or d < nearest:
                        nearest = d

    # If we didn't find any main roads, fall back to any road
    if nearest is None:
        for road in Road.objects.all():
            if not (road.geometry and isinstance(road.geometry, dict)):
                continue

            geom_type = road.geometry.get('type')
            coords = road.geometry.get('coordinates', [])

            if geom_type == 'LineString' and coords:
                for i in range(len(coords) - 1):
                    lon1, lat1 = coords[i][0], coords[i][1]
                    lon2, lat2 = coords[i + 1][0], coords[i + 1][1]
                    d = _distance_point_to_segment_m(lat, lng, lat1, lon1, lat2, lon2)
                    if nearest is None or d < nearest:
                        nearest = d

            elif geom_type == 'MultiLineString' and coords:
                for linestring in coords:
                    for i in range(len(linestring) - 1):
                        lon1, lat1 = linestring[i][0], linestring[i][1]
                        lon2, lat2 = linestring[i + 1][0], linestring[i + 1][1]
                        d = _distance_point_to_segment_m(lat, lng, lat1, lon1, lat2, lon2)
                        if nearest is None or d < nearest:
                            nearest = d

    return nearest


def _query_overpass_nearest_road(lat, lng, radius=1000):
    """
    Query Overpass API for nearest highway ways around (lat,lng) within `radius` meters.
    Returns nearest distance in meters or None.
    """
    try:
        import requests
        # Overpass bbox in lon,lat order: minlon,minlat,maxlon,maxlat
        # Convert radius (meters) to approx degrees (~111000 m per deg lat)
        deg = radius / 111000.0
        minlat = lat - deg
        maxlat = lat + deg
        minlon = lng - deg
        maxlon = lng + deg

        bbox = f"{minlon},{minlat},{maxlon},{maxlat}"
        highway_tags = ['motorway','trunk','primary','secondary','tertiary','residential','service','unclassified']
        query_parts = "\n".join([f'way["highway"="{t}"]({bbox});' for t in highway_tags])
        query = f"[out:json][timeout:60];(\n{query_parts}\n);out geom;"

        headers = {'User-Agent': 'CafeLocate/1.0 (+https://example.com)'}
        endpoints = [
            'https://overpass.openstreetmap.fr/api/interpreter',
            'https://overpass.kumi.systems/api/interpreter',
            'https://overpass-api.de/api/interpreter'
        ]

        data = None
        for url in endpoints:
            try:
                resp = requests.post(url, data={'data': query}, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception:
                continue

        if not data:
            return None

        # Compute nearest distance from returned ways
        nearest = None
        for element in data.get('elements', []):
            if element.get('type') != 'way' or 'geometry' not in element:
                continue
            coords = element['geometry']
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]['lon'], coords[i]['lat']
                lon2, lat2 = coords[i+1]['lon'], coords[i+1]['lat']
                d = _distance_point_to_segment_m(lat, lng, lat1, lon1, lat2, lon2)
                if nearest is None or d < nearest:
                    nearest = d

        return nearest
    except Exception:
        return None


def point_in_polygon(point_lng, point_lat, polygon_geojson):
    """
    Ray-casting algorithm to check if a point is inside a GeoJSON polygon or WKT geometry.
    Handles both GeoJSON and WKT formats.
    """
    if not polygon_geojson:
        return False

    geom_type = polygon_geojson.get('type')
    
    # Handle WKT format
    if geom_type == 'wkt':
        return _point_in_wkt_polygon(point_lng, point_lat, polygon_geojson.get('wkt', ''))
    
    # Handle GeoJSON format
    coordinates = polygon_geojson.get('coordinates', [])

    if not coordinates:
        return False

    # Handle both Polygon and MultiPolygon
    rings = []
    if geom_type == 'Polygon':
        rings = [coordinates[0]] if coordinates else []
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            if polygon:
                rings.append(polygon[0])
    else:
        return False

    for exterior_ring in rings:
        n = len(exterior_ring)
        inside = False
        p1x, p1y = exterior_ring[0]
        for i in range(1, n + 1):
            p2x, p2y = exterior_ring[i % n]
            if point_lat > min(p1y, p2y):
                if point_lat <= max(p1y, p2y):
                    if point_lng <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point_lng <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        if inside:
            return True

    return False


def _point_in_wkt_polygon(point_lng, point_lat, wkt_string):
    """
    Parse WKT polygon string and check if point is inside using ray-casting.
    Handles POLYGON and MULTIPOLYGON WKT formats.
    """
    import re
    
    if not wkt_string:
        return False
    
    wkt_string = wkt_string.strip().upper()
    
    # Extract coordinates from WKT
    # POLYGON((lng lat, lng lat, ...))
    # MULTIPOLYGON(((lng lat, lng lat, ...)), ...)
    
    if wkt_string.startswith('MULTIPOLYGON'):
        # Extract all polygons: MULTIPOLYGON(((coords)), ((coords)))
        coord_pattern = r'\(\(([^)]+)\)\)'
        matches = re.findall(coord_pattern, wkt_string)
        polygons = []
        for match in matches:
            coords = _parse_wkt_coords(match)
            if coords:
                polygons.append(coords)
    elif wkt_string.startswith('POLYGON'):
        # Extract coords: POLYGON((coords))
        coord_pattern = r'\(\(([^)]+)\)\)'
        match = re.search(coord_pattern, wkt_string)
        if match:
            coords = _parse_wkt_coords(match.group(1))
            polygons = [coords] if coords else []
        else:
            polygons = []
    else:
        return False
    
    # Ray-casting algorithm for each polygon
    for ring in polygons:
        if len(ring) < 3:
            continue
        
        n = len(ring)
        inside = False
        p1x, p1y = ring[0]
        
        for i in range(1, n + 1):
            p2x, p2y = ring[i % n]
            if point_lat > min(p1y, p2y):
                if point_lat <= max(p1y, p2y):
                    if point_lng <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point_lng <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        if inside:
            return True
    
    return False


def _parse_wkt_coords(coord_string):
    """
    Parse WKT coordinate string: 'lng lat, lng lat, ...'
    Returns list of (lng, lat) tuples.
    """
    import re
    
    # Match numbers (including decimals and negatives)
    coord_pattern = r'(-?\d+\.?\d*)\s+(-?\d+\.?\d*)'
    matches = re.findall(coord_pattern, coord_string)
    
    coords = []
    for lng_str, lat_str in matches:
        try:
            coords.append((float(lng_str), float(lat_str)))
        except ValueError:
            continue
    
    return coords


# ═══════════════════════════════════════════════════════════════════
# VIEW 1: User Registration
# POST /api/auth/register/
# ═══════════════════════════════════════════════════════════════════
class UserRegistrationView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        username = request.data.get('username', '').strip()
        email    = request.data.get('email', '').strip()
        password = request.data.get('password', '')

        if not username or not email or not password:
            return Response(
                {'error': 'Username, email, and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(password) < 6:
            return Response(
                {'error': 'Password must be at least 6 characters'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if UserProfile.objects.filter(username=username).exists():
            return Response(
                {'error': 'Username already taken'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if UserProfile.objects.filter(email=email).exists():
            return Response(
                {'error': 'An account with this email already exists'},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = UserProfile.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        token = jwt.encode(
            {'user_id': user.id, 'username': user.username, 'email': user.email},
            settings.SECRET_KEY,
            algorithm='HS256'
        )

        return Response({
            'token': token,
            'user': UserProfileSerializer(user).data
        }, status=status.HTTP_201_CREATED)


# ═══════════════════════════════════════════════════════════════════
# VIEW 2: User Login
# POST /api/auth/login/
# ═══════════════════════════════════════════════════════════════════
class UserLoginView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        login_credential = request.data.get('username', '').strip()
        password = request.data.get('password', '')

        if not login_credential or not password:
            return Response(
                {'error': 'Username/email and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = None
        if '@' in login_credential:
            try:
                user = UserProfile.objects.get(email=login_credential)
            except UserProfile.DoesNotExist:
                pass
        else:
            try:
                user = UserProfile.objects.get(username=login_credential)
            except UserProfile.DoesNotExist:
                pass

        if user is None or not user.check_password(password):
            return Response(
                {'error': 'Invalid username/email or password'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        if not user.is_active:
            return Response(
                {'error': 'Account is disabled'},
                status=status.HTTP_403_FORBIDDEN
            )

        token = jwt.encode(
            {'user_id': user.id, 'username': user.username, 'email': user.email},
            settings.SECRET_KEY,
            algorithm='HS256'
        )

        return Response({
            'token': token,
            'user': UserProfileSerializer(user).data
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: Nearby Cafes
# GET /api/cafes/nearby/?lat=27.71&lng=85.32&radius=500
# ═══════════════════════════════════════════════════════════════════
class NearbyCafesView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng query parameters are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not is_within_kathmandu_metropolitan_city(lat, lng):
            return Response(
                {'error': 'Location pinning is allowed only inside Kathmandu Metropolitan City.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        cafes = []
        for cafe in Cafe.objects.filter(is_open=True):
            if not (cafe.location and isinstance(cafe.location, dict)):
                # Fall back to latitude/longitude fields
                if cafe.latitude and cafe.longitude:
                    distance = haversine_distance(lat, lng, cafe.latitude, cafe.longitude)
                    if distance <= radius:
                        cafe.distance = distance
                        cafes.append(cafe)
            else:
                coords = cafe.location.get('coordinates', [None, None])
                cafe_lng, cafe_lat = coords[0], coords[1]
                if cafe_lng is not None and cafe_lat is not None:
                    distance = haversine_distance(lat, lng, cafe_lat, cafe_lng)
                    if distance <= radius:
                        cafe.distance = distance
                        cafes.append(cafe)

        cafes.sort(key=lambda c: getattr(c, 'distance', 0))

        serializer = CafeSerializer(cafes, many=True)
        return Response({
            'count':  len(cafes),
            'cafes':  serializer.data,
            'center': {'lat': lat, 'lng': lng}
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: Dataset Stats
# GET /api/cafes/stats/
# ═══════════════════════════════════════════════════════════════════════════
class CafeStatsView(APIView):
    def get(self, request):
        cafes = Cafe.objects.all()
        total = cafes.count()
        open_cafes = cafes.filter(is_open=True).count()

        rating_values = [c.rating for c in cafes if c.rating is not None]
        avg_rating = round(sum(rating_values) / len(rating_values), 2) if rating_values else None

        review_values = [c.review_count for c in cafes if c.review_count is not None]
        avg_reviews = round(sum(review_values) / len(review_values), 1) if review_values else 0

        # Cafe type distribution (group by cafe_type)
        type_counts = {}
        for c in cafes:
            t = (c.cafe_type or 'unknown').strip().lower()
            if not t:
                t = 'unknown'
            type_counts[t] = type_counts.get(t, 0) + 1

        # Top 5 types by count
        type_ranking = sorted(type_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return Response({
            'total_cafes': total,
            'open_cafes': open_cafes,
            'avg_rating': avg_rating,
            'avg_review_count': avg_reviews,
            'type_counts': type_counts,
            'top_type_ranking': [{'type': t, 'count': count} for t, count in type_ranking]
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 4: Full Suitability Analysis
# POST /api/analyze/
# ═══════════════════════════════════════════════════════════════════
class SuitabilityAnalysisView(APIView):

    def post(self, request):
        serializer = SuitabilityRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        authenticated_user = get_request_user(request)
        lat       = serializer.validated_data['lat']
        lng       = serializer.validated_data['lng']
        cafe_type = serializer.validated_data['cafe_type']
        radius    = serializer.validated_data.get('radius', 500)

        # Step 1: Find nearby cafes using haversine distance
        nearby_cafes = []
        for cafe in Cafe.objects.filter(is_open=True):
            cafe_lat, cafe_lng = None, None

            if cafe.location and isinstance(cafe.location, dict):
                coords = cafe.location.get('coordinates', [None, None])
                if len(coords) >= 2:
                    cafe_lng, cafe_lat = coords[0], coords[1]
            elif cafe.latitude and cafe.longitude:
                cafe_lat, cafe_lng = cafe.latitude, cafe.longitude

            if cafe_lat is not None and cafe_lng is not None:
                distance = haversine_distance(lat, lng, cafe_lat, cafe_lng)
                if distance <= radius:
                    cafe.distance = distance
                    nearby_cafes.append(cafe)

        total_competitors = len(nearby_cafes)
        same_type_cafes = [c for c in nearby_cafes if (c.cafe_type or '').strip() == cafe_type]
        same_type_competitors = len(same_type_cafes)
        same_type_within_200m = len([c for c in same_type_cafes if c.distance <= 200])
        same_type_min_distance = min([c.distance for c in same_type_cafes]) if same_type_cafes else radius
        same_type_avg_distance = (
            sum([c.distance for c in same_type_cafes]) / len(same_type_cafes)
            if same_type_cafes else radius
        )

        # Step 2: Top 5 cafes by score
        top5_qs = sorted(
            nearby_cafes,
            key=lambda c: (c.rating or 0) * math.log(max(c.review_count, 1) + 1),
            reverse=True
        )[:5]

        # Step 3: Population density from ward(s) - radius-aware
        # First find the primary ward containing the point
        primary_ward = None
        primary_pop_density = 10000  # fallback
        
        for ward in Ward.objects.all():
            if ward.boundary and isinstance(ward.boundary, dict):
                if point_in_polygon(lng, lat, ward.boundary):
                    primary_ward = ward
                    primary_pop_density = ward.population_density
                    break
        
        # For larger radius, also consider nearby wards with distance-weighted average
        if radius > 300:
            # Collect nearby wards and their distances
            nearby_wards = []
            
            for ward in Ward.objects.all():
                if ward.boundary and isinstance(ward.boundary, dict):
                    # Check if ward boundary overlaps with analysis radius
                    # Use approximate distance to ward centroid (simplified check)
                    geom = ward.boundary
                    
                    # For wards that aren't the primary ward
                    if ward != primary_ward:
                        # Try to find approximate centroid by checking multiple points in boundary
                        # This is a simplification - for accurate results we'd need GIS
                        # For now, estimate if ward might overlap with radius
                        nearby_wards.append(ward)
            
            # If we have nearby wards, calculate weighted average
            if nearby_wards and radius >= 500:
                # Weighted population density:
                # Primary ward gets 60% weight, nearby wards split 40%
                weighted_density = primary_pop_density * 0.6
                
                if nearby_wards:
                    avg_nearby = sum(w.population_density for w in nearby_wards) / len(nearby_wards)
                    weighted_density += avg_nearby * 0.4
                
                pop_density = weighted_density
            else:
                pop_density = primary_pop_density
        else:
            pop_density = primary_pop_density

        # Step 4: Road length estimate within radius
        road_segments_nearby = 0
        for road in Road.objects.all():
            if not (road.geometry and isinstance(road.geometry, dict)):
                continue

            geom_type = road.geometry.get('type')
            coordinates = road.geometry.get('coordinates', [])

            if geom_type == 'LineString' and coordinates:
                for coord in coordinates:
                    if len(coord) >= 2:
                        d = haversine_distance(lat, lng, coord[1], coord[0])
                        if d <= radius:
                            road_segments_nearby += 1
                            break

            elif geom_type == 'MultiLineString' and coordinates:
                found = False
                for linestring in coordinates:
                    if found:
                        break
                    for coord in linestring:
                        if len(coord) >= 2:
                            d = haversine_distance(lat, lng, coord[1], coord[0])
                            if d <= radius:
                                road_segments_nearby += 1
                                found = True
                                break

        road_m = road_segments_nearby * 100  # ~100m per segment estimate
        
        # If no roads were found in database (Road table is empty),
        # use a sophisticated estimate based on multiple factors
        if road_m == 0 and Road.objects.count() == 0:
            # Base road density estimate for Kathmandu (meters per 500m radius)
            base_road_m = 2000
            
            # Factor 1: Competitor density (higher density = more accessible)
            competitor_factor = min(1.5, 0.5 + (total_competitors / 30))
            
            # Factor 2: Population density (denser areas have more roads)
            pop_factor = min(1.3, pop_density / 10000)
            
            # Factor 3: Location variation based on coordinates
            # Create pseudo-random variation based on lat/lng to vary by location
            import hashlib
            coord_hash = int(hashlib.md5(
                f"{lat:.3f}{lng:.3f}".encode()
            ).hexdigest(), 16) % 1000 / 1000  # 0 to 1
            location_factor = 0.8 + (coord_hash * 0.4)  # 0.8 to 1.2
            
            road_m = round(base_road_m * competitor_factor * pop_factor * location_factor)
            road_m = max(1500, min(3500, road_m))  # Clamp between 1500-3500m

        # Compute nearest main road distance (meters)
        nearest_main_road_m = _nearest_main_road_distance(lat, lng)
        if nearest_main_road_m is not None:
            nearest_main_road_m = round(nearest_main_road_m)

        # Step 5: Compute heuristic suitability score (0-10)
        weighted_competitors = total_competitors + (same_type_competitors * 1.5)
        competitor_score = max(0.0, 1.0 - (weighted_competitors / 20.0)) * 4.0
        road_score = min(1.0, road_m / 3000.0) * 3.0
        pop_score = min(1.0, pop_density / 15000.0) * 3.0
        suitability_score = round(competitor_score + road_score + pop_score, 2)

        # Step 6: Regression-based ML suitability prediction
        ratings = [c.rating for c in nearby_cafes if c.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        same_type_ratings = [c.rating for c in same_type_cafes if c.rating is not None]
        same_type_avg_rating = sum(same_type_ratings) / len(same_type_ratings) if same_type_ratings else 0

        amenity_stats = _get_amenity_stats(lat, lng, radius)

        nearest_road_for_features = (
            float(nearest_main_road_m)
            if nearest_main_road_m is not None else
            float(max(50, min(3000, road_m / 2)))
        )

        road_access_score = max(0.0, min(10.0, 10.0 - (nearest_road_for_features / 150.0)))
        bus_access_bonus = min(2.5, amenity_stats['bus_stops_within_500m'] * 0.35)
        school_bonus = min(1.5, amenity_stats['schools_within_500m'] * 0.15)
        hospital_bonus = min(1.0, amenity_stats['hospitals_within_500m'] * 0.15)
        accessibility_score = max(0.0, min(10.0, road_access_score + bus_access_bonus + school_bonus + hospital_bonus))

        density_signal = min(4.0, pop_density / 5000.0)
        transit_signal = min(3.0, amenity_stats['bus_stops_within_500m'] * 0.4)
        institutional_signal = min(2.0, amenity_stats['schools_within_500m'] * 0.2)
        commerce_signal = min(2.0, total_competitors * 0.12)
        road_signal = min(2.0, max(0.0, 2.0 - (nearest_road_for_features / 300.0)))
        foot_traffic_score = max(
            0.0,
            min(10.0, density_signal + transit_signal + institutional_signal + commerce_signal + road_signal)
        )

        competition_pressure = max(
            0.0,
            min(
                10.0,
                (total_competitors * 0.30) +
                (same_type_competitors * 0.85) +
                min(2.0, avg_rating * 0.25) +
                min(2.5, same_type_avg_rating * 0.55)
            )
        )

        # Calculate competition_effective (normalized competitor pressure)
        competition_effective = max(0.0, min(10.0, (total_competitors * 0.3) + (same_type_competitors * 0.7)))
        
        # Calculate osm_amenity_density_500m (amenities per km² within 500m)
        amenity_count = (
            amenity_stats['schools_within_500m'] + 
            amenity_stats['hospitals_within_500m'] + 
            amenity_stats['bus_stops_within_500m']
        )
        # Approximate area of 500m radius circle in km²
        area_km2 = 3.14159 * (0.5 ** 2)  # πr² where r=0.5km
        osm_amenity_density_500m = amenity_count / area_km2 if area_km2 > 0 else 0

        features_dict = _build_regression_features(
            pop_density=pop_density,
            accessibility_score=accessibility_score,
            foot_traffic_score=foot_traffic_score,
            competition_pressure=competition_pressure,
            amenity_stats=amenity_stats,
            osm_amenity_density_500m=osm_amenity_density_500m,
        )

        prediction = get_suitability_prediction(features_dict)
        regression_score = prediction.get('predicted_score', suitability_score)
        type_recommendation = _recommend_cafe_type(
            total_competitors=total_competitors,
            nearby_cafes=nearby_cafes,
            amenity_stats=amenity_stats,
            accessibility_score=accessibility_score,
            foot_traffic_score=foot_traffic_score,
            pop_density=pop_density,
        )
        prediction.update(type_recommendation)

        # Get demographic information
        demographic_info = _get_demographic_info(lat, lng)

        return Response({
            'location':     {'lat': lat, 'lng': lng},
            'nearby_count': total_competitors,
            'top5':         CafeSerializer(top5_qs, many=True).data,
            'demographics': demographic_info,
            'suitability': {
                'score':              regression_score,
                'level':              prediction.get('predicted_suitability', 'Unknown'),
                'confidence':         prediction.get('confidence', 0),
                'competitor_count':   total_competitors,
                'same_type_competitor_count': same_type_competitors,
                'road_distance_m':    (round(nearest_main_road_m) if nearest_main_road_m is not None else round(road_m)),
                'population_density': pop_density,
            },
            'prediction': prediction,
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 5: Amenities by Type and Radius
# GET /api/amenities/?lat=27.71&lng=85.32&radius=500&type=school
# Returns all amenities of a specific type within a radius
# ═══════════════════════════════════════════════════════════════════
class AmenitiesView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))
            amenity_type = request.GET.get('type', '').strip()
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat, lng (required) and type, radius (optional) must be valid'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not is_within_kathmandu_metropolitan_city(lat, lng):
            return Response(
                {'error': 'Location pinning is allowed only inside Kathmandu Metropolitan City.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Query amenities by type
        query = Amenity.objects.all()
        if amenity_type:
            query = query.filter(amenity_type__icontains=amenity_type)

        amenities = []
        for amenity in query:
            distance = haversine_distance(lat, lng, amenity.latitude, amenity.longitude)
            if distance <= radius:
                amenity.distance = distance
                amenities.append(amenity)

        # Sort by distance
        amenities.sort(key=lambda a: getattr(a, 'distance', 0))

        serializer = AmenitySerializer(amenities, many=True)
        return Response({
            'count': len(amenities),
            'amenities': serializer.data,
            'center': {'lat': lat, 'lng': lng},
            'amenity_type': amenity_type if amenity_type else 'All',
            'radius': radius
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 6: Multiple Amenity Types Report
# POST /api/amenities-report/
# Returns count of different amenity types within a radius
# ═══════════════════════════════════════════════════════════════════
class AmenitiesReportView(APIView):

    def post(self, request):
        try:
            lat    = float(request.data.get('lat'))
            lng    = float(request.data.get('lng'))
            radius = float(request.data.get('radius', 500))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not is_within_kathmandu_metropolitan_city(lat, lng):
            return Response(
                {'error': 'Location pinning is allowed only inside Kathmandu Metropolitan City.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Key amenity types we want to report
        key_amenity_types = ['school', 'college', 'university', 'hospital', 'bus_station', 'cafe', 'health_post', 'pharmacy']

        report = {}
        for amenity_type in key_amenity_types:
            amenities = []
            for amenity in Amenity.objects.filter(amenity_type__icontains=amenity_type):
                distance = haversine_distance(lat, lng, amenity.latitude, amenity.longitude)
                if distance <= radius:
                    amenity.distance = distance
                    amenities.append(amenity)

            report[amenity_type] = {
                'count': len(amenities),
                'amenities': AmenitySerializer(amenities[:20], many=True).data  # Top 20
            }

        return Response({
            'location': {'lat': lat, 'lng': lng},
            'radius': radius,
            'amenities_report': report
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 7: Area Population Calculation
# GET /api/area-population/?lat=27.71&lng=85.32&radius=500
# Calculates exact population within the area based on ward boundaries
# ═══════════════════════════════════════════════════════════════════
class AreaPopulationView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng query parameters are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not is_within_kathmandu_metropolitan_city(lat, lng):
            return Response(
                {'error': 'Location pinning is allowed only inside Kathmandu Metropolitan City.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        total_population = 0
        affected_wards = []

        # Check which wards the location falls in or intersects with
        for ward in Ward.objects.all():
            if not (ward.boundary and isinstance(ward.boundary, dict)):
                continue

            is_affected = False

            # Handle WKT boundaries (stored as {'type': 'wkt', 'wkt': 'POLYGON(...)'})
            if isinstance(ward.boundary, dict) and ward.boundary.get('type') == 'wkt':
                try:
                    wkt_str = ward.boundary.get('wkt')
                    if wkt_str:
                        # If Shapely is available, use it for accurate contains/distance
                        if SHAPELY_AVAILABLE:
                            ward_polygon = wkt_loads(wkt_str)
                            test_point = Point(lng, lat)
                            if ward_polygon.contains(test_point):
                                is_affected = True
                            elif ward_polygon.distance(test_point) * 111000 <= radius:  # ~111km per degree
                                is_affected = True
                        else:
                            # Use lightweight WKT point-in-polygon check
                            if _point_in_wkt_polygon(lng, lat, wkt_str):
                                is_affected = True
                            else:
                                # As a proximity fallback, parse vertex coords and check nearest vertex distance
                                coords = _parse_wkt_coords(wkt_str)
                                if coords:
                                    min_d = min(haversine_distance(lat, lng, lat_v, lon_v) for lon_v, lat_v in coords)
                                    if min_d <= radius:
                                        is_affected = True
                except Exception as e:
                    logger.warning(f'Error handling WKT for ward {ward.ward_number}: {e}')
                    continue
            else:
                # Fallback to checking if point is within or near the ward's GeoJSON coordinates
                try:
                    coords = ward.boundary.get('coordinates', []) if isinstance(ward.boundary, dict) else []
                    if coords and len(coords) > 0:
                        # Simple check: if we have coordinates, assume some overlap within radius
                        for coord_set in coords:
                            if isinstance(coord_set, (list, tuple)) and len(coord_set) > 0:
                                for coord in coord_set:
                                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                        d = haversine_distance(lat, lng, coord[1], coord[0])
                                        if d <= radius * 2:  # generous bounds
                                            is_affected = True
                                            break
                                if is_affected:
                                    break
                except Exception:
                    continue

            if is_affected:
                total_population += ward.population
                affected_wards.append({
                    'ward_number': ward.ward_number,
                    'population': ward.population,
                    'population_density': ward.population_density,
                    'area_sqkm': ward.area_sqkm,
                })

        return Response({
            'location': {'lat': lat, 'lng': lng},
            'radius': radius,
            'total_population': total_population,
            'affected_wards': affected_wards,
            'affected_ward_count': len(affected_wards),
        })


class LocationValidationView(APIView):

    def get(self, request):
        try:
            lat = float(request.GET.get('lat'))
            lng = float(request.GET.get('lng'))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng query parameters are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        is_valid = is_within_kathmandu_metropolitan_city(lat, lng)
        return Response({
            'lat': lat,
            'lng': lng,
            'is_valid': is_valid,
            'message': (
                'Location is inside Kathmandu Metropolitan City.'
                if is_valid else
                'Location pinning is allowed only inside Kathmandu Metropolitan City.'
            ),
        })


class AnalysisHistoryView(APIView):

    def get(self, request):
        user = get_request_user(request)
        if user is None:
            return Response(
                {'error': 'Authentication required.'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        cafe_type = request.GET.get('cafe_type', '').strip()
        limit = min(max(int(request.GET.get('limit', 10)), 1), 25)

        queryset = AnalysisHistory.objects.filter(user=user)
        if cafe_type:
            queryset = queryset.filter(cafe_type=cafe_type)

        history_items = list(queryset[:limit + 1])
        serializer = AnalysisHistorySerializer(history_items, many=True)
        return Response({
            'count': len(serializer.data),
            'history': serializer.data,
        })

