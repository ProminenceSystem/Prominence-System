"""
The Pytheas API Interface
Provides REST API endpoints for accessing The Pytheas platform functionality
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from datetime import datetime

from ..core.periplus import Periplus
from ..market_analysis.analyzer import MarketAnalyzer
from ..blockchain.solana_client import SolanaClient

app = FastAPI(
    title="The Pytheas API",
    description="AI-Powered Market Navigation System for Solana Memetic Markets",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
periplus = Periplus()
market_analyzer = MarketAnalyzer()
solana_client = SolanaClient()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "The Pytheas API",
        "version": "0.1.0",
        "status": "operational"
    }

@app.get("/market/analysis/{token_address}")
async def get_market_analysis(token_address: str):
    """Get comprehensive market analysis for a token"""
    try:
        # Gather token information
        token_info = await solana_client.get_token_info(token_address)
        
        # Analyze market data
        market_data = await market_analyzer.analyze_market_depth({
            "token_address": token_address,
            "token_info": token_info
        })
        
        # Generate AI insights
        ai_insights = periplus.generate_navigation_signals(
            market_data=market_data,
            sentiment_data={},  # Placeholder
            risk_profile="moderate"
        )
        
        return {
            "token_info": token_info,
            "market_analysis": market_data,
            "ai_insights": ai_insights,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/social/sentiment/{token_address}")
async def get_social_sentiment(token_address: str):
    """Get social media sentiment analysis for a token"""
    try:
        social_data = market_analyzer.analyze_social_signals({
            "token_address": token_address,
            "timeframe": "24h"
        })
        
        return {
            "token_address": token_address,
            "sentiment_analysis": social_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/whale/tracking/{token_address}")
async def track_whale_movements(token_address: str):
    """Track whale wallet movements for a token"""
    try:
        transactions = await solana_client.analyze_token_transactions(
            token_address,
            limit=1000
        )
        
        whale_analysis = market_analyzer.detect_whale_movements(
            transactions.get("transactions", [])
        )
        
        return {
            "token_address": token_address,
            "whale_analysis": whale_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/signals/{token_address}")
async def get_trading_signals(token_address: str, risk_profile: str = "moderate"):
    """Get AI-generated trading signals for a token"""
    try:
        # Gather all necessary data
        token_info = await solana_client.get_token_info(token_address)
        market_data = await market_analyzer.analyze_market_depth({"token_address": token_address})
        social_data = market_analyzer.analyze_social_signals({"token_address": token_address})
        
        # Generate signals
        signals = periplus.generate_navigation_signals(
            market_data=market_data,
            sentiment_data=social_data,
            risk_profile=risk_profile
        )
        
        return {
            "token_address": token_address,
            "signals": signals,
            "supporting_data": {
                "market_data": market_data,
                "social_data": social_data
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 