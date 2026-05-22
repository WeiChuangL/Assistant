import asyncio
import json

import httpx

from src.agent.tool import register_tool


@register_tool(
    name="get_weather",
    description="查询指定城市当前的实时天气信息，包括温度、天气状况、湿度、风速",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如 '北京'、'Beijing'、'Tokyo'",
            }
        },
        "required": ["city"],
    },
)
async def get_weather(city: str) -> str:
    """Query weather via wttr.in (free, no API key needed)."""
    url = f"https://wttr.in/{city}?format=j1"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "N/A")
        desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
        humidity = current.get("humidity", "N/A")
        windspeed = current.get("windspeedKmph", "N/A")
        feels_like = current.get("FeelsLikeC", "N/A")

        # Get city name from request
        location = ""
        for n in data.get("nearest_area", []):
            name_parts = [
                n.get("areaName", [{}])[0].get("value", ""),
                n.get("region", [{}])[0].get("value", ""),
                n.get("country", [{}])[0].get("value", ""),
            ]
            location = ", ".join(p for p in name_parts if p)
            break

        return json.dumps({
            "城市": city,
            "地点": location or city,
            "温度": f"{temp_c}°C",
            "体感温度": f"{feels_like}°C",
            "天气": desc,
            "湿度": f"{humidity}%",
            "风速": f"{windspeed} km/h",
        }, ensure_ascii=False, indent=2)
    except httpx.HTTPError as e:
        return f"天气查询失败: {e}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"天气数据解析失败: {e}"
