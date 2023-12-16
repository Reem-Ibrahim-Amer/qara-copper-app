import streamlit as st
st.set_page_config(page_title="Calculator", page_icon="📈")

def calculate_price(mass, unit, rate_per_kg=5.0):
    # Assuming a rate of $5 per kilogram
    if unit == "Kilograms":
        price = mass * rate_per_kg
    elif unit == "Pounds":
        # Convert pounds to kilograms (1 pound = 0.453592 kg)
        price = mass * rate_per_kg / 0.453592
    else:
        st.error("Invalid unit")
        return None

    return price

st.markdown("""<center><h1 style="color:#FC5E22">Today's Copper Price Calculator</h2></center>""", unsafe_allow_html=True)

# User inputs
mass = st.number_input("Enter the mass:", min_value=0.0, step=0.1)
unit = st.selectbox("Select unit:", ["Kilograms", "Pounds"])


# Calculate price
if st.button("Calculate Price"):
    price = calculate_price(mass, unit)
    if price is not None:
        st.success(f"The calculated price is: ${price:.5f}")
