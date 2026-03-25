// CafeSuitabilityAnalysis - API Module
// Handles all communication with the Django backend

class APIManager {
    constructor() {
        this.baseURL = 'http://localhost:8000/api';
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add auth token if available
        if (window.authManager && window.authManager.getToken()) {
            config.headers['Authorization'] = `Bearer ${window.authManager.getToken()}`;
        }

        const response = await fetch(url, config);

        if (!response.ok) {
            let errMsg = `Server error (${response.status})`;
            try {
                const errData = await response.json();
                errMsg = errData.error || errData.detail || errMsg;
            } catch (_) {}
            throw new Error(errMsg);
        }

        return await response.json();
    }

    /**
     * Get nearby cafes within radius
     */
    async getNearbyCafes(lat, lng, radius = 500) {
        return this.makeRequest(`/cafes/nearby/?lat=${lat}&lng=${lng}&radius=${radius}`);
    }

    /**
     * Get full suitability analysis (main endpoint)
     */
    async getSuitabilityAnalysis(lat, lng, cafeType, radius = 500) {
        return this.makeRequest('/analyze/', {
            method: 'POST',
            body: JSON.stringify({
                lat: lat,
                lng: lng,
                cafe_type: cafeType,
                radius: radius
            })
        });
    }

    async validateLocation(lat, lng) {
        return this.makeRequest(`/validate-location/?lat=${lat}&lng=${lng}`);
    }

    async getAnalysisHistory(cafeType, limit = 10) {
        const params = new URLSearchParams({ limit: String(limit) });
        if (cafeType) params.set('cafe_type', cafeType);
        return this.makeRequest(`/history/?${params.toString()}`);
    }

    /**
     * Get global cafe dataset statistics
     */
    async getCafeDatasetStats() {
        return this.makeRequest('/cafes/stats/');
    }

    /**
     * Get ML model predictions (Random Forest + XGBoost)
     */
    async getMLPrediction(lat, lng, cafeType, radius = 500) {
        return this.makeRequest('/ml-prediction/', {
            method: 'POST',
            body: JSON.stringify({
                lat: lat,
                lng: lng,
                cafe_type: cafeType,
                radius: radius
            })
        });
    }

    /**
     * Format errors for user display
     */
    formatError(error) {
        const msg = error.message || '';
        if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
            return 'Cannot connect to server. Make sure the backend is running on port 8000.';
        }
        if (msg.includes('401')) return 'Session expired. Please log in again.';
        if (msg.includes('500')) return 'Server error. Please try again later.';
        return msg || 'An unexpected error occurred.';
    }
}

// Initialize API manager
document.addEventListener('DOMContentLoaded', () => {
    window.apiManager = new APIManager();
});
