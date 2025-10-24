<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class RoomPredictionService
{
    protected $apiBaseUrl;
    protected $timeout;

    public function __construct()
    {
        // Get from config or .env file
        $this->apiBaseUrl = env('FASTAPI_URL', 'http://127.0.0.1:8000');
        $this->timeout = env('FASTAPI_TIMEOUT', 30);
    }

    /**
     * Check FastAPI health status
     */
    public function checkHealth(): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->get($this->apiBaseUrl . '/health');

            if ($response->successful()) {
                return $response->json();
            }

            return ['status' => 'error', 'message' => 'FastAPI service unavailable'];
        } catch (\Exception $e) {
            Log::error('FastAPI health check failed: ' . $e->getMessage());
            return ['status' => 'error', 'message' => $e->getMessage()];
        }
    }

    /**
     * Predict room based on booking features
     */
    public function predictRoom(array $features): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post($this->apiBaseUrl . '/predict', [
                    'department' => $features['department'],
                    'duration_hours' => (float) $features['duration_hours'],
                    'event_period' => $features['event_period'],
                    'seats' => (int) $features['seats']
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            throw new \Exception('FastAPI prediction failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('Room prediction failed: ' . $e->getMessage());
            throw $e;
        }
    }

    /**
     * Get room prediction probabilities
     */
    public function predictRoomProbabilities(array $features): array
    {
        try {
            $response = Http::timeout($this->timeout)
                ->post($this->apiBaseUrl . '/predict_proba', [
                    'department' => $features['department'],
                    'duration_hours' => (float) $features['duration_hours'],
                    'event_period' => $features['event_period'],
                    'seats' => (int) $features['seats']
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            throw new \Exception('FastAPI probability prediction failed: ' . $response->body());
        } catch (\Exception $e) {
            Log::error('Room probability prediction failed: ' . $e->getMessage());
            throw $e;
        }
    }

    /**
     * Get raw prediction with additional metadata
     */
    public function getPredictionWithMetadata(array $features): array
    {
        try {
            // Get basic prediction
            $prediction = $this->predictRoom($features);
            
            // Try to get probabilities (may not be supported by all models)
            $probabilities = null;
            try {
                $probaResult = $this->predictRoomProbabilities($features);
                $probabilities = $probaResult['probabilities'] ?? null;
            } catch (\Exception $e) {
                // Model might not support probabilities
                Log::info('Model does not support probability predictions');
            }

            return [
                'room' => $prediction['room'],
                'probabilities' => $probabilities,
                'request_features' => $features,
                'prediction_time' => now()->toISOString()
            ];
        } catch (\Exception $e) {
            throw $e;
        }
    }
}