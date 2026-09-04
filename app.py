import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)
st.title(" House Price Prediction System")
st.write("Predict the estimated price of a house using its features.")


@st.cache_data
def load_data():
    return pd.read_csv("house_data.csv")


df = load_data()


X = df.drop(columns=["price", "house_id"])
y = df["price"]

categorical_features = ["neighborhood"]

numeric_features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "stories",
    "garage_spaces",
    "age_years",
    "distance_to_center_km",
    "has_pool",
    "has_garden",
    "school_rating",
    "crime_index",
    "renovated"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)



predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


st.subheader(" Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Houses",
        len(df)
    )

with col2:
    st.metric(
        "Average Price",
        f"{df['price'].mean():,.0f}"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2 * 100:.2f}%"
    )


st.divider()


st.subheader(" Predict House Price")

col1, col2 = st.columns(2)


with col1:

    area_sqft = st.number_input(
        "Area (sqft)",
        min_value=300,
        max_value=10000,
        value=1800
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    stories = st.number_input(
        "Stories",
        min_value=1,
        max_value=5,
        value=2
    )

    garage_spaces = st.number_input(
        "Garage Spaces",
        min_value=0,
        max_value=6,
        value=2
    )

    age_years = st.number_input(
        "House Age (Years)",
        min_value=0,
        max_value=150,
        value=20
    )


with col2:

    distance_to_center_km = st.number_input(
        "Distance to Center (km)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        sorted(df["neighborhood"].unique())
    )

    has_pool = st.selectbox(
        "Has Pool?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    has_garden = st.selectbox(
        "Has Garden?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    school_rating = st.number_input(
        "School Rating",
        min_value=1,
        max_value=10,
        value=5
    )

    crime_index = st.number_input(
        "Crime Index",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    renovated = st.selectbox(
        "Renovated?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


if st.button(
    " Predict House Price",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "area_sqft": [area_sqft],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "garage_spaces": [garage_spaces],
        "age_years": [age_years],
        "distance_to_center_km": [distance_to_center_km],
        "neighborhood": [neighborhood],
        "has_pool": [has_pool],
        "has_garden": [has_garden],
        "school_rating": [school_rating],
        "crime_index": [crime_index],
        "renovated": [renovated]
    })

    predicted_price = model.predict(input_data)[0]

    st.divider()

    st.success(" House Price Prediction")

    st.metric(
        "Estimated House Price",
        f"{predicted_price:,.0f}"
    )

    st.info(
        "The estimated price is generated using a "
        "Random Forest machine learning model."
    )



st.divider()

st.subheader(" Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Mean Absolute Error",
        f"{mae:,.0f}"
    )

with col2:
    st.metric(
        "R² Score",
        f"{r2 * 100:.2f}%"
    )

st.divider()

st.subheader(" House Dataset")

st.dataframe(
    df,
    use_container_width=True
)
