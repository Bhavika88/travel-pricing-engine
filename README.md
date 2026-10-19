Luxury villa rentals (think Airbnb Luxe, Onefinestay, Plum Guide) face a classic dilemma:

Price too high → Villa sits empty (lost revenue).

Price too low → Villa is booked instantly but you left money on the table.

The Goal: Build a system that recommends the optimal nightly price for a luxury villa on any given date, by predicting two things:

Demand: How many bookings will we get at a given price?

Willingness to Pay: What is the maximum price the market will bear?

Data Design Philosophy: The dataset embeds 5 realistic patterns (seasonality, lead-time effect, luxury amenity premiums, event-driven demand, and price elasticity). This allows the downstream models to discover these patterns rather than being told them — which mirrors how real-world ML systems extract value from noisy business data.