<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\RoomRecommendationController;

/*
|--------------------------------------------------------------------------
| API Routes for Room Recommendation Service
|--------------------------------------------------------------------------
*/

// Health check endpoint
Route::get('/health', [RoomRecommendationController::class, 'health']);

// Room prediction endpoints
Route::post('/predict', [RoomRecommendationController::class, 'predict']);
Route::post('/predict-proba', [RoomRecommendationController::class, 'predictProba']);

// Get available rooms list
Route::get('/rooms', [RoomRecommendationController::class, 'getRooms']);