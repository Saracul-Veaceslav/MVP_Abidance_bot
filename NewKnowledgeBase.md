# New Knowledge Base

This document tracks new insights and knowledge gained while working on the Abidance Crypto Trading Bot.

## Project Structure Insights

- The project follows a clean architecture approach with clearly defined boundaries between components
- Each component (exchange, strategy, risk, data, cli) is isolated in its own package to promote separation of concerns
- Test-driven development (TDD) approach is being used to ensure code quality and reliability

## Technical Learnings

- CCXT library provides a unified API for interacting with multiple cryptocurrency exchanges
- Implementing trading strategies requires careful handling of decimal precision to avoid rounding errors
- Position sizing and risk management are critical components of a successful trading system 