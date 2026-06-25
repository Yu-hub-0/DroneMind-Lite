import json
import math


DRONE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_no_fly_zone",
            "description": "查询指定地理坐标是否处于无人机禁飞区或限飞区",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度"},
                    "longitude": {"type": "number", "description": "经度"}
                },
                "required": ["latitude", "longitude"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_for_flight",
            "description": "获取飞行气象条件，包括风速、能见度、降水概率和飞行建议",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "位置名称或坐标"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_flight_time",
            "description": "根据电池容量、电量、重量和风速估算剩余飞行时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "battery_capacity_mah": {"type": "number"},
                    "battery_percentage": {"type": "number"},
                    "drone_weight_kg": {"type": "number"},
                    "wind_speed_ms": {"type": "number", "default": 0}
                },
                "required": ["battery_capacity_mah", "battery_percentage", "drone_weight_kg"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_mission_route",
            "description": "根据测区面积、高度和重叠率估算航测航线参数",
            "parameters": {
                "type": "object",
                "properties": {
                    "area_km2": {"type": "number"},
                    "overlap_rate": {"type": "number", "default": 0.75},
                    "side_overlap_rate": {"type": "number", "default": 0.65},
                    "flight_altitude_m": {"type": "number"},
                    "sensor_width_mm": {"type": "number", "default": 13.2},
                    "focal_length_mm": {"type": "number", "default": 8.8}
                },
                "required": ["area_km2", "flight_altitude_m"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_drone_specs",
            "description": "查询常见无人机机型的简要参数",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "机型名称"}
                },
                "required": ["model"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_uav_units",
            "description": "无人机常用单位换算",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]


def check_no_fly_zone(args):
    lat = float(args.get("latitude", 0))
    lon = float(args.get("longitude", 0))
    near_beijing_core = 39.80 <= lat <= 40.05 and 116.25 <= lon <= 116.55
    return {
        "status": "restricted" if near_beijing_core else "unknown",
        "risk_level": "high" if near_beijing_core else "medium",
        "message": "示例规则：北京核心区附近按高风险限飞处理，实际飞行前必须查询官方空域平台。",
        "latitude": lat,
        "longitude": lon
    }


def get_weather_for_flight(args):
    location = args.get("location", "unknown")
    samples = {
        "北京": {"wind_speed_ms": 5.5, "visibility_km": 8, "rain_probability": 0.1},
        "上海": {"wind_speed_ms": 7.8, "visibility_km": 6, "rain_probability": 0.2},
        "深圳": {"wind_speed_ms": 4.2, "visibility_km": 10, "rain_probability": 0.35}
    }
    weather = next((v for k, v in samples.items() if k in location), {"wind_speed_ms": 6.0, "visibility_km": 8, "rain_probability": 0.15})
    suitable = weather["wind_speed_ms"] < 8 and weather["visibility_km"] >= 5 and weather["rain_probability"] < 0.3
    return {"location": location, "suitable": suitable, "advice": "可谨慎飞行" if suitable else "不建议飞行", **weather}


def calculate_flight_time(args):
    capacity = float(args.get("battery_capacity_mah", 0))
    percentage = float(args.get("battery_percentage", 0))
    weight = max(float(args.get("drone_weight_kg", 1)), 0.1)
    wind = max(float(args.get("wind_speed_ms", 0)), 0)
    base_minutes = capacity / 220.0 * (percentage / 100.0)
    penalty = 1 + max(weight - 1.0, 0) * 0.18 + wind * 0.035
    minutes = max(base_minutes / penalty, 0)
    return {"estimated_minutes": round(minutes, 1), "reserve_minutes": 5, "usable_minutes": round(max(minutes - 5, 0), 1)}


def plan_mission_route(args):
    area = float(args.get("area_km2", 0))
    altitude = max(float(args.get("flight_altitude_m", 100)), 1)
    overlap = float(args.get("overlap_rate", 0.75))
    side_overlap = float(args.get("side_overlap_rate", 0.65))
    footprint_width_m = altitude * 0.75
    line_spacing_m = max(footprint_width_m * (1 - side_overlap), 5)
    route_length_km = area * 1000 / line_spacing_m if line_spacing_m else 0
    photo_interval_m = max(altitude * 0.55 * (1 - overlap), 3)
    return {
        "line_spacing_m": round(line_spacing_m, 1),
        "route_length_km": round(route_length_km, 2),
        "photo_interval_m": round(photo_interval_m, 1),
        "recommended_speed_ms": 5,
        "note": "估算结果用于训练/演示，实际任务需结合相机参数、地形和法规复核。"
    }


def lookup_drone_specs(args):
    model = args.get("model", "").lower()
    specs = {
        "mavic 3": {"max_flight_time_min": 46, "wind_resistance_ms": 12, "use_case": "航拍/轻量测绘"},
        "mini 4 pro": {"max_flight_time_min": 34, "wind_resistance_ms": 10.7, "use_case": "入门航拍"},
        "phantom 4 rtk": {"max_flight_time_min": 30, "wind_resistance_ms": 10, "use_case": "测绘"},
        "matrice 350": {"max_flight_time_min": 55, "wind_resistance_ms": 12, "use_case": "行业巡检"}
    }
    for key, value in specs.items():
        if key in model:
            return {"model": key, **value}
    return {"model": args.get("model", "unknown"), "message": "未命中内置样例机型，可接入真实机型数据库。"}


def convert_uav_units(args):
    value = float(args.get("value", 0))
    from_unit = args.get("from_unit", "").lower()
    to_unit = args.get("to_unit", "").lower()
    rates = {
        ("m/s", "km/h"): 3.6,
        ("km/h", "m/s"): 1 / 3.6,
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
        ("km2", "ha"): 100,
        ("ha", "km2"): 0.01,
        ("mah", "ah"): 0.001,
        ("ah", "mah"): 1000
    }
    rate = rates.get((from_unit, to_unit))
    if rate is None:
        return {"error": f"unsupported conversion: {from_unit} -> {to_unit}"}
    return {"value": round(value * rate, 4), "unit": to_unit}


TOOL_FUNCTIONS = {
    "check_no_fly_zone": check_no_fly_zone,
    "get_weather_for_flight": get_weather_for_flight,
    "calculate_flight_time": calculate_flight_time,
    "plan_mission_route": plan_mission_route,
    "lookup_drone_specs": lookup_drone_specs,
    "convert_uav_units": convert_uav_units,
    # Backward-compatible names used by early Agent samples.
    "get_drone_specs": lookup_drone_specs,
    "convert_drone_units": convert_uav_units,
}


def execute_tool(name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    fn = TOOL_FUNCTIONS.get(name)
    if not fn:
        return {"error": f"unknown tool: {name}"}
    return fn(arguments or {})


if __name__ == "__main__":
    print(json.dumps(DRONE_TOOLS, ensure_ascii=False, indent=2))
