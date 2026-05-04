# US Accidents Visualization Dashboard

This repository contains an updated and reworked version of a university data visualization project about US traffic accidents from 2016 to 2023.

The original project was created at Aarhus University by:

- Piotr Eliasz
- Ali Al Mais
- Ibrahim Ahmed Mohammed Haras
- Xiaoyang Zhang

Original project date: December 16, 2024.

This version was later updated and improved by **Piotr Eliasz**.  
The goal of the rework was to clean up the project structure, improve usability, simplify running the app, and make the dashboard more practical as an interactive analytical tool.

## What this project is

This project is an interactive dashboard for exploring accident data in the United States.

It is built to help inspect:

- where accidents happen
- how accident severity is distributed
- how road features relate to accidents
- how weather conditions relate to severity
- how accident counts change over time

The dashboard is intended for exploratory analysis rather than static reporting.

## Main visualizations

The dashboard combines several linked visualizations:

- **Accident map**  
  A Plotly-based map showing accident locations across the United States.  
  Points are colored by severity and the map supports interaction and filtering.

- **Road features and severity by state**  
  A grid of state-level choropleth maps showing accident counts for selected road features such as:
  `Crossing`, `Give_Way`, `Junction`, `Stop`, and `Traffic_Signal`.  
  The grid is organized by severity level, which makes it possible to compare how road features and severity vary across states.

- **Weather condition and severity distribution**  
  A stacked bar chart showing how severity is distributed within each weather condition group.  
  This makes it easier to compare whether some weather categories are more associated with higher-severity accidents.

- **Monthly accident trend by severity**  
  A time-based chart showing monthly accident counts split by severity level.  
  This helps reveal trends, seasonality, and changes across years.

## Interaction

The visualizations are linked.

This means the dashboard can be filtered interactively by selecting values from the visualizations, for example:

- selecting a state from the map
- selecting a state and road-feature row from the choropleth grid
- selecting a weather condition
- selecting a month or time range

There is also a reset button to clear active filters and return to the default view.

## Technologies

- Python
- Dash
- Plotly
- Polars / Pandas
- Flask-Caching
- Docker

## Running with Docker

Run:

```bash
docker compose up --build
```

The Docker setup will:

1. download the dataset automatically if it is missing
2. start the dashboard

Then open:

```text
http://localhost:8050
```
