<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Services\RoomPredictionService;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Validator;

class RoomRecommendationController extends Controller
{
    protected $roomService;

    public function __construct(RoomPredictionService $roomService)
    {
        $this->roomService = $roomService;
    }

    /**
     * Health check endpoint
     */
    public function health(): JsonResponse
    {
        $fastApiStatus = $this->roomService->checkHealth();
        
        return response()->json([
            'status' => 'ok',
            'fastapi_status' => $fastApiStatus,
            'timestamp' => now()->toISOString()
        ]);
    }

    /**
     * Predict room based on booking features
     */
    public function predict(Request $request): JsonResponse
    {
        // Validate input
        $validator = Validator::make($request->all(), [
            'department' => 'required|string',
            'duration_hours' => 'required|numeric|min:0',
            'event_period' => 'required|string|in:เช้า,บ่าย,เย็น',
            'seats' => 'required|integer|min:1'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'error' => 'Validation failed',
                'messages' => $validator->errors()
            ], 422);
        }

        try {
            $result = $this->roomService->predictRoom($request->all());
            
            return response()->json([
                'success' => true,
                'room' => $result['room'],
                'timestamp' => now()->toISOString()
            ]);
            
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Prediction failed',
                'message' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get room prediction probabilities
     */
    public function predictProba(Request $request): JsonResponse
    {
        // Validate input
        $validator = Validator::make($request->all(), [
            'department' => 'required|string',
            'duration_hours' => 'required|numeric|min:0',
            'event_period' => 'required|string|in:เช้า,บ่าย,เย็น',
            'seats' => 'required|integer|min:1'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'error' => 'Validation failed',
                'messages' => $validator->errors()
            ], 422);
        }

        try {
            $result = $this->roomService->predictRoomProbabilities($request->all());
            
            return response()->json([
                'success' => true,
                'probabilities' => $result['probabilities'],
                'timestamp' => now()->toISOString()
            ]);
            
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Prediction failed',
                'message' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get list of available rooms
     */
    public function getRooms(): JsonResponse
    {
        // This could be from database or configuration
        $rooms = [
            'ห้องประชุม 1',
            'ห้องประชุม 2',
            'ห้องประชุม 3',
            'ห้องประชุมใหญ่',
            'ห้องประชุมเล็ก',
            'ห้องประชุม VIP'
        ];

        return response()->json([
            'rooms' => $rooms,
            'count' => count($rooms)
        ]);
    }
}