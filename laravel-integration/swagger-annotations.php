<?php

/**
 * @OA\Info(
 *     title="Room Recommendation API (Laravel)",
 *     version="1.0.0",
 *     description="Laravel API wrapper for FastAPI Room Recommendation Service",
 *     @OA\Contact(
 *         email="admin@example.com"
 *     )
 * )
 * 
 * @OA\Server(
 *     url="http://localhost:8000/api",
 *     description="Local development server"
 * )
 * 
 * @OA\Schema(
 *     schema="BookingFeatures",
 *     type="object",
 *     required={"department", "duration_hours", "event_period", "seats"},
 *     @OA\Property(property="department", type="string", example="งานบริหารทั่วไป"),
 *     @OA\Property(property="duration_hours", type="number", format="float", example=2.5),
 *     @OA\Property(property="event_period", type="string", enum={"เช้า", "บ่าย", "เย็น"}, example="บ่าย"),
 *     @OA\Property(property="seats", type="integer", minimum=1, example=35)
 * )
 * 
 * @OA\Schema(
 *     schema="PredictionResponse",
 *     type="object",
 *     @OA\Property(property="success", type="boolean", example=true),
 *     @OA\Property(property="room", type="string", example="ห้องประชุม 1"),
 *     @OA\Property(property="timestamp", type="string", format="date-time")
 * )
 * 
 * @OA\Schema(
 *     schema="ProbabilityResponse",
 *     type="object",
 *     @OA\Property(property="success", type="boolean", example=true),
 *     @OA\Property(
 *         property="probabilities",
 *         type="object",
 *         example={"ห้องประชุม 1": 0.7, "ห้องประชุม 2": 0.2, "ห้องประชุม 3": 0.1}
 *     ),
 *     @OA\Property(property="timestamp", type="string", format="date-time")
 * )
 */

/**
 * @OA\Get(
 *     path="/health",
 *     tags={"Health"},
 *     summary="Check API and FastAPI service health",
 *     @OA\Response(
 *         response=200,
 *         description="Service is healthy",
 *         @OA\JsonContent(
 *             @OA\Property(property="status", type="string", example="ok"),
 *             @OA\Property(property="fastapi_status", type="object"),
 *             @OA\Property(property="timestamp", type="string", format="date-time")
 *         )
 *     )
 * )
 */

/**
 * @OA\Post(
 *     path="/predict",
 *     tags={"Prediction"},
 *     summary="Predict room based on booking features",
 *     @OA\RequestBody(
 *         required=true,
 *         @OA\JsonContent(ref="#/components/schemas/BookingFeatures")
 *     ),
 *     @OA\Response(
 *         response=200,
 *         description="Room predicted successfully",
 *         @OA\JsonContent(ref="#/components/schemas/PredictionResponse")
 *     ),
 *     @OA\Response(
 *         response=422,
 *         description="Validation error"
 *     ),
 *     @OA\Response(
 *         response=500,
 *         description="Prediction failed"
 *     )
 * )
 */

/**
 * @OA\Post(
 *     path="/predict-proba",
 *     tags={"Prediction"},
 *     summary="Get room prediction probabilities",
 *     @OA\RequestBody(
 *         required=true,
 *         @OA\JsonContent(ref="#/components/schemas/BookingFeatures")
 *     ),
 *     @OA\Response(
 *         response=200,
 *         description="Probabilities retrieved successfully",
 *         @OA\JsonContent(ref="#/components/schemas/ProbabilityResponse")
 *     ),
 *     @OA\Response(
 *         response=422,
 *         description="Validation error"
 *     ),
 *     @OA\Response(
 *         response=500,
 *         description="Prediction failed"
 *     )
 * )
 */

/**
 * @OA\Get(
 *     path="/rooms",
 *     tags={"Rooms"},
 *     summary="Get list of available rooms",
 *     @OA\Response(
 *         response=200,
 *         description="List of rooms",
 *         @OA\JsonContent(
 *             @OA\Property(property="rooms", type="array", @OA\Items(type="string")),
 *             @OA\Property(property="count", type="integer")
 *         )
 *     )
 * )
 */