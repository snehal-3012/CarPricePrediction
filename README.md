# Car Price Prediction Model

## Overview

This repository contains a Python script that predicts the MSRP (Manufacturer's Suggested Retail Price) of cars based on various features. The dataset used for training the model includes information about the car's make, model, engine specifications, and more. The model uses linear regression to predict car prices.

## Dataset

The dataset used in this project is `car_features_and_msrm.csv`, which includes the following features:

- **Year**: The year the car was manufactured.
- **Make**: The manufacturer of the car.
- **Model**: The specific model of the car.
- **Engine HP**: Horsepower of the car's engine.
- **Engine Cylinders**: Number of cylinders in the car's engine.
- **Transmission Type**: The type of transmission (e.g., automatic, manual).
- **Driven Wheels**: The drivetrain of the car (e.g., FWD, RWD, AWD).
- **Number of Doors**: The number of doors on the car.
- **Market Category**: The market segment the car belongs to.
- **Vehicle Size**: The size category of the vehicle (e.g., compact, midsize).
- **Vehicle Style**: The style of the vehicle (e.g., sedan, SUV).
- **Highway MPG**: The car's fuel efficiency on the highway.
- **City MPG**: The car's fuel efficiency in the city.
- **MSRP**: The target variable representing the car's price.
- **Popularity**: The popularity representing the brand value of car.
- **Engine Fuel Type**: This parameter represents type of fuel that car need.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/CarPricePrediction.git
cd CarPricePrediction
pip install -r requirements.txt
```

## Usage

### Run the Model:

The main script is CarPricePrediction.py. It reads the dataset, preprocesses the data, trains a linear regression model, and evaluates its performance.

To run the script, simply execute:
```
python CarPricePrediction.py
```

## Developer Information

- **Name**: Snehal Yadav
- **Contact**: [snehalyadav3099@gmail.com](mailto:snehalyadav3099@gmail.com)
