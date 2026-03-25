from django.urls import path
from . import views

urlpatterns = [
    # POST /api/auth/register/
    # Create new user account with username, email, password
    path('auth/register/', views.UserRegistrationView.as_view(), name='user-register'),

    # POST /api/auth/login/
    # Authenticate user with username/email and password
    path('auth/login/',    views.UserLoginView.as_view(),    name='user-login'),

    # GET /api/cafes/nearby/?lat=27.71&lng=85.32&radius=500
    # Returns list of all cafes within radius meters of point
    path('cafes/nearby/', views.NearbyCafesView.as_view(), name='cafes-nearby'),

    # POST /api/analyze/
    # Main analysis: nearby + top5 + suitability score + ML prediction
    path('analyze/',      views.SuitabilityAnalysisView.as_view(), name='analyze'),

    # GET /api/amenities/?lat=27.71&lng=85.32&radius=500&type=school
    # Returns amenities (schools, hospitals, bus stops, etc.) within radius
    path('amenities/', views.AmenitiesView.as_view(), name='amenities'),

    # POST /api/amenities-report/
    # Returns count of different amenity types (school, hospital, bus_station) within radius
    path('amenities-report/', views.AmenitiesReportView.as_view(), name='amenities-report'),

    # GET /api/area-population/?lat=27.71&lng=85.32&radius=500
    # Calculates exact population within the area based on ward boundaries
    path('area-population/', views.AreaPopulationView.as_view(), name='area-population'),

    # GET /api/cafes/stats/
    # Returns dataset level statistics for all cafes
    path('cafes/stats/', views.CafeStatsView.as_view(), name='cafes-stats'),

    # GET /api/validate-location/?lat=27.71&lng=85.32
    # Returns whether a point is inside Kathmandu Metropolitan City
    path('validate-location/', views.LocationValidationView.as_view(), name='validate-location'),

    # GET /api/history/?cafe_type=coffee_shop
    # Returns saved analysis history for logged-in users
    path('history/', views.AnalysisHistoryView.as_view(), name='analysis-history'),
    
    # POST /api/ml-prediction/
    # ML-based cafe suitability prediction using Random Forest + XGBoost ensemble
    path('ml-prediction/', views.MLSuitabilityPredictionView.as_view(), name='ml-prediction'),
]

# Final URL structure:
#   /api/auth/register/   ← user registration
#   /api/auth/login/      ← user login
#   /api/cafes/nearby/    ← nearby cafes within radius
#   /api/analyze/         ← main suitability analysis
#   /api/amenities/       ← amenities by type within radius
#   /api/amenities-report/ ← summary report of key amenities
#   /api/area-population/ ← exact population for area
