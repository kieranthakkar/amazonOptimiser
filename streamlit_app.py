# Import all modules
import pandas as pd, numpy as np, streamlit as st, pickle
from sklearn.feature_extraction import FeatureHasher
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load the trained model
@st.cache_data()
def load_model_LR():
    pickle_in = open("Models/linear_regression.pkl", "rb")
    model = pickle.load(pickle_in)
    pickle_in.close()
    return model

# Categories for use in a drop-down list
unique_categories = ['Hi-Fi Speakers', 'CD, Disc & Tape Players', 'Wearable Technology', 'Light Bulbs', 'Bathroom Lighting', 'Heating, Cooling & Air Quality', 'Coffee & Espresso Machines', 'Lab & Scientific Products',
 'Smart Speakers', 'Motorbike Clothing', 'Motorbike Accessories', 'Motorbike Batteries', 'Motorbike Boots & Luggage', 'Motorbike Chassis', 'Handmade Home & Kitchen Products', 'Hardware',
 'Storage & Home Organisation', 'Fireplaces, Stoves & Accessories', 'PC Gaming Accessories', 'USB Gadgets', 'Blank Media Cases & Wallets', 'Car & Motorbike', 'Boys', 'Sports & Outdoors', 'Microphones',
 'String Instruments', 'Karaoke Equipment', 'PA & Stage', 'General Music-Making Accessories', 'Wind Instruments', 'Handmade Gifts', 'Fragrances', 'Calendars & Personal Organisers', 'Furniture & Lighting', 'Computer Printers',
 'Ski Goggles', 'Snowboards', 'Skiing Poles', 'Downhill Ski Boots', 'Hiking Hand & Foot Warmers', 'Pet Supplies', 'Plants, Seeds & Bulbs', 'Garden Furniture & Accessories', 'Bird & Wildlife Care',
 'Storage & Organisation', 'Living Room Furniture', 'Bedding & Linen', 'Curtain & Blind Accessories', 'Skin Care', "Kids' Art & Craft Supplies", "Kids' Play Vehicles", 'Hobbies', 'Laptops', 'Projectors',
 'Graphics Cards', 'Computer Memory', 'Motherboards', 'Power Supplies', 'CPUs', 'Computer Screws', 'Streaming Clients', '3D Printers', 'Barebone PCs', "Women's Sports & Outdoor Shoes", 'Luxury Food & Drink',
 'Alexa Built-In Devices', 'PC & Video Games', 'SIM Cards', 'Mobile Phone Accessories', 'Birthday Gifts', 'Handmade Kitchen & Dining', 'Abrasive & Finishing Products', 'Professional Medical Supplies',
 'Cutting Tools', 'Material Handling Products', 'Packaging & Shipping Supplies', 'Power & Hand Tools', 'Agricultural Equipment & Supplies', 'Tennis Shoes', 'Boating Footwear', 'Cycling Shoes', 'Bath & Body',
 'Home Brewing & Wine Making', 'Tableware', 'Kitchen Storage & Organisation', 'Kitchen Tools & Gadgets', 'Cookware', 'Water Coolers, Filters & Cartridges', 'Beer, Wine & Spirits', 'Manicure & Pedicure Products',
 'Flashes', 'Computers, Components & Accessories', 'Home Audio Record Players', 'Radios & Boomboxes', 'Car & Vehicle Electronics', 'eBook Readers & Accessories', 'Lighting', 'Small Kitchen Appliances',
 'Motorbike Engines & Engine Parts', 'Motorbike Drive & Gears', 'Motorbike Brakes', 'Motorbike Exhaust & Exhaust Systems', 'Motorbike Handlebars, Controls & Grips', 'Mowers & Outdoor Power Tools',
 'Kitchen & Bath Fixtures', 'Rough Plumbing', 'Monitor Accessories', 'Cables & Accessories', 'Guitars & Gear', 'Pens, Pencils & Writing Supplies', 'School & Educational Supplies', 'Ski Clothing',
 'Outdoor Heaters & Fire Pits', 'Garden Décor', 'Beauty', 'Made in Italy Handmade', 'Cushions & Accessories', 'Home Fragrance', 'Window Treatments', 'Home Entertainment Furniture', 'Dining Room Furniture',
 'Home Bar Furniture', 'Kitchen Linen', 'Mattress Pads & Toppers', "Children's Bedding", 'Bedding Accessories', 'Games & Game Accessories', 'Dolls & Accessories', 'Sports Toys & Outdoor', 'Monitors', 'I/O Port Cards',
 'Computer Cases', 'KVM Switches', 'Printers & Accessories', 'Telephones, VoIP & Accessories', 'Handmade Artwork', 'Industrial Electrical', 'Test & Measurement', '3D Printing & Scanning', 'Basketball Footwear', 'Make-up',
 'Surveillance Cameras', 'Photo Printers', 'Tripods & Monopods', 'Mobile Phones & Communication', 'Electrical Power Accessories', 'Radio Communication', 'Outdoor Rope Lights', 'Vacuums & Floorcare',
 'Large Appliances', 'Motorbike Lighting', 'Motorbike Seat Covers', 'Motorbike Instruments', 'Motorbike Electrical & Batteries', 'Lights and switches', 'Plugs', 'Home Entertainment', 'Girls',
 'Painting Supplies, Tools & Wall Treatments', 'Building Supplies', 'Safety & Security', 'Tablet Accessories', 'Keyboards, Mice & Input Devices', 'Laptop Accessories', 'Headphones & Earphones', 'Baby',
 'Smartwatches', 'Piano & Keyboard', 'Drums & Percussion', 'Synthesisers, Samplers & Digital Instruments', 'Office Electronics', 'Office Supplies', 'Gardening', 'Outdoor Cooking', 'Decking & Fencing',
 'Thermometers & Meteorological Instruments', 'Pools, Hot Tubs & Supplies', 'Health & Personal Care', 'Decorative Artificial Flora', 'Candles & Holders', 'Signs & Plaques', 'Home Office Furniture',
 'Bathroom Furniture', 'Inflatable Beds, Pillows & Accessories', 'Bathroom Linen', 'Bedding Collections', "Kids' Play Figures", 'Baby & Toddler Toys', 'Learning & Education Toys', 'Toy Advent Calendars',
 'Electronic Toys', 'Tablets', 'External Sound Cards', 'Internal TV Tuner & Video Capture Cards', 'External TV Tuners & Video Capture Cards', 'Scanners & Accessories', "Men's Sports & Outdoor Shoes",
 'Darts & Dartboards', 'Table Tennis', 'Billiard, Snooker & Pool', 'Bowling', 'Trampolines & Accessories', 'Handmade Clothing, Shoes & Accessories', 'Handmade Home Décor', 'Handmade', 'Smart Home Security & Lighting',
 'Professional Education Supplies', 'Hydraulics, Pneumatics & Plumbing', 'Ballet & Dancing Footwear', 'Cricket Shoes', 'Golf Shoes', 'Boxing Shoes', 'Men', 'Headphones, Earphones & Accessories', 'Bakeware',
 'Grocery', 'Lenses', 'Camcorders', 'Camera & Photo Accessories', 'Household Batteries, Chargers & Accessories', 'Home Cinema, TV & Video', 'Hi-Fi & Home Audio Accessories', 'Portable Sound & Video Products', 'Outdoor Lighting',
 'Torches', 'Sports Supplements', 'Ironing & Steamers', "Customers' Most Loved", 'Cameras', 'Electrical', 'Construction Machinery', 'Handmade Baby Products', 'USB Hubs', 'Computer Audio & Video Accessories',
 'Adapters', 'Computer & Server Racks', 'Hard Drive Accessories', 'Printer Accessories', 'Computer Memory Card Accessories', 'Uninterruptible Power Supply Units & Accessories', 'Luggage and travel gear', 'Bass Guitars & Gear', 'Recording & Computer',
 'DJ & VJ Equipment', 'Art & Craft Supplies', 'Office Paper Products', 'Ski Helmets', 'Snowboard Boots', 'Snowboard Bindings', 'Downhill Skis', 'Snow Sledding Equipment', 'Networking Devices', 'Garden Storage & Housing',
 'Garden Tools & Watering Equipment', 'Photo Frames', 'Rugs, Pads & Protectors', 'Mirrors', 'Clocks', 'Doormats', 'Decorative Home Accessories', 'Boxes & Organisers', 'Slipcovers', 'Vases', 'Bedroom Furniture', 'Hallway Furniture',
 'Jigsaws & Puzzles', 'Building & Construction Toys', 'Remote & App-Controlled Devices', "Kids' Dress Up & Pretend Play", 'Soft Toys', 'Desktop PCs', 'External Optical Drives', 'Internal Optical Drives', 'Network Cards', 'Data Storage',
 'Mobile Phones & Smartphones', 'Handmade Jewellery', 'Gifts for Him', 'Gifts for Her', 'Women', 'Hockey Shoes', 'Climbing Footwear', 'Equestrian Sports Boots', 'Arts & Crafts', 'Hair Care', 'Coffee, Tea & Espresso',
 'Digital Cameras', 'Digital Frames', 'Action Cameras', 'Film Cameras', 'Binoculars, Telescopes & Optics', 'Media Streaming Devices', 'Hi-Fi Receivers & Separates', 'GPS, Finders & Accessories', 'Indoor Lighting', 'String Lights'
 ]
unique_categories.sort()

model_w2v = Word2Vec.load(r"Models/word2vec_model.model")

# FeatureHasher for categoryName
n_features = len(unique_categories)
categories = [[category] for category in unique_categories]
hasher = FeatureHasher(n_features=n_features, input_type="string")
X_category = hasher.transform(categories).toarray().astype("float16")
hashed_df = pd.DataFrame(X_category, columns=[f"hash_{i}" for i in range(n_features)])

# Word2Vec function
def get_word_vectors(product_name):
    try:
        five_words = word_tokenize(product_name.lower())[:5]
        vectors = [model_w2v.wv[word] for word in five_words]
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.nan
    except KeyError:
        return np.nan

def main():
    st.title("Amazon Price Predictor")

    LR = load_model_LR()

    # User input
    reviews = st.number_input("Number of Reviews", min_value=0)
    rating = st.slider("Rating", min_value=1.0, max_value=5.0, step=0.1, value=4.0)
    isBestSeller = st.checkbox("Is Best Seller")
    title = st.text_input("Product Title")
    category = st.selectbox("Product Category", unique_categories)
    boughtInLastMonth = st.number_input("boughtInLastMonth", min_value=0)

    # Button to trigger prediction -- DOES IT WORK? OR IS IT AUTOMATIC?
    if st.button("Predict Price"):
        # Preprocess user input
        isBestSeller = 1 if isBestSeller else 0
        category_hash = hasher.transform([[category]]).toarray().astype("float16").flatten()
        category_hash = np.pad(category_hash, (0, 296 - len(category_hash)))  # Ensure it has 296 columns
        
        title_vector = get_word_vectors(title)
        title_vector_expanded = np.zeros(model_w2v.vector_size)
        for i in range(model_w2v.vector_size):
            if isinstance(title_vector, np.ndarray):
                title_vector_expanded[i] = title_vector[i]

        # Prepare input for prediction
        user_input = {
        'stars': rating,
        'reviews': reviews,
        'isBestSeller': isBestSeller,
        "boughtInLastMonth": boughtInLastMonth,
        'hashed_category': category_hash
        }

        user_input.update(dict(zip([f'vector_dim_{i + 1}' for i in range(model_w2v.vector_size)], title_vector_expanded)))

        # Convert user_input to a numpy array
        input_array = np.concatenate([np.array([value]) if not isinstance(value, np.ndarray) else value for value in user_input.values()])

        # Transpose
        input_array = input_array.reshape(1, -1)

        # Make prediction
        predicted_price_LR = LR.predict(input_array)
        predicted_price_LR = np.expm1(predicted_price_LR)[0]

        # Display the prediction
        st.subheader("Predicted Price:")
        st.write(f"LinearRegression: £{predicted_price_LR:.2f}")

if __name__ == "__main__":
    main()