import streamlit as st
from passlib.hash import pbkdf2_sha256
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Database Configuration
DATABASE_URL = r"sqlite:///C:\Users\caran\OneDrive\Desktop\python\mydatabase.db"
engine = create_engine(DATABASE_URL)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

Base.metadata.create_all(engine)

# Sign-up Page
def signup():
    st.title("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match")
            return

        try:
            # Save user info to the database
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            db = SessionLocal()
            new_user = User(username=email, email=email, password=pbkdf2_sha256.hash(password))
            db.add(new_user)
            db.commit()
            db.close()
            st.success("Sign up successful. You can now log in.")
        except Exception as e:
            st.error(f"Error: {e}")

# Login Page
def login():
    with st.form(key="login_form"):
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            try:
                # Authenticate user
                SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
                db = SessionLocal()
                user = db.query(User).filter(User.username == username).first()
                db.close()
                if user and pbkdf2_sha256.verify(password, user.password):
                    st.success(f"Logged in as {username}")
                    st.session_state["username"] = username
                else:
                    st.error("Invalid username or password")
            except Exception as e:
                st.error(f"Error: {e}")

# Main Content
def main():
    if "username" in st.session_state:
        st.write(f'Welcome {st.session_state["username"]}')
        st.title("Machine Learning Studio by Team STRAW_HAT")
        st.subheader("Welcome to the Machine Learning Studio!")
        st.markdown('''Build. Train. Deploy. Machine Learning Made Easy.
        
        Drag, drop, and build powerful ML models. Our intuitive studio empowers everyone to harness the power of data.  Start your journey today!''')

        # Sidebar
        st.sidebar.subheader("Setup your Problem Here")
        problem_type = st.sidebar.selectbox("Pick your Problem Type", ["Regression", "Classification", "Clustering", "Image Classification"])

        if problem_type == "Regression":
            state = 1
        elif problem_type == "Classification":
            state = 2
        elif problem_type == "Clustering":
            state = 3
        else:
            state = 4

        if state == 4:
            img_zip_file = st.sidebar.file_uploader("Upload your Dataset", type=['zip'])
        else:
            dataset_file = st.sidebar.file_uploader("Upload your Dataset", type=['csv'])

        if state != 4:
            if dataset_file:
                st.subheader("Your Dataset:-")
                df = pd.read_csv(dataset_file)
                st.dataframe(df)
                if state == 1 or state == 2:
                    target_y = st.sidebar.text_input("Enter the Target Variable Column Name (Leave Blank to use Last Column)")

        train_btn = st.sidebar.button("Train")

        if st.session_state.get('button') != True:
            st.session_state['button'] = train_btn

        if st.session_state['button'] == True:
            comp_table_flag = 1

            if state != 4:
                if (state == 1 or state == 2) and target_y != "":
                    cols = list(df.columns.values)
                    cols.pop(cols.index(target_y))
                    df = df[cols+[target_y]]

                model = Models(df)
                model.clean_and_scale_dataset()

                if state == 1:
                    regression_models = ["Linear Regression", "Decision Tree Regression", "SVR", "Ridge Regression", "Lasso Regression",
                                        "ElasticNet", "Random Forest Regressor", "Multi-Layer Perceptron", "KNN Regressor", 
                                        "Gradient Boosting Regressor"]
                    lr = list(model.linear_regression())
                    dtr = list(model.dtree_regressor())
                    svr = list(model.SVR())
                    rr = list(model.ridge_regression())
                    lsr = list(model.lasso())
                    en = list(model.elasticnet())
                    rfr = list(model.random_forest_regressor())
                    mlpr = list(model.mlp_regressor())
                    knnr = list(model.knn_regressor())
                    gbr = list(model.gradient_boost_regressor())

                    regression_funcs = {
                        "Linear Regression": lr[0],
                        "Decision Tree Regression": dtr[0],
                        "SVR": svr[0],
                        "Ridge Regression": rr[0],
                        "Lasso Regression": lsr[0],
                        "ElasticNet": en[0],
                        "Random Forest Regressor": rfr[0],
                        "Multi-Layer Perceptron": mlpr[0],
                        "KNN Regressor": knnr[0],
                        "Gradient Boosting Regressor": gbr[0]
                    } 

                    metrics = list(lr[1].keys())

                    regressors_table = {
                        "Linear Regression":
                        list(lr[1].values()),
                        "Decision Tree Regression": list(dtr[1].values()),
                        "SVR": list(svr[1].values()),
                        "Ridge Regression": list(rr[1].values()),
                        "Lasso Regression": list(lsr[1].values()),
                        "ElasticNet": list(en[1].values()),
                        "Random Forest Regressor": list(rfr[1].values()),
                        "Multi-Layer Perceptron": list(mlpr[1].values()),
                        "KNN Regressor": list(knnr[1].values()),
                        "Gradient Boosting Regressor": list(gbr[1].values())
                    }
                    comp_table = pd.DataFrame.from_dict(regressors_table)
                    comp_table.index = metrics
                    comp_table = comp_table.transpose()

                elif state == 2:
                    classification_models = ["SVM", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", 
                                            "AdaBoost", "Multi-Layer Perceptron", "Gradient Boosting", "Random Forest"]
                    s = list(model.SVM())
                    lor = list(model.logistic_regression())
                    dt = list(model.dtree_classifier())
                    knn = list(model.knn_classifier())
                    nb = list(model.naivebayes())
                    ab = list(model.adaboost())
                    mlp = list(model.mlp())
                    gb = list(model.gradient_boost())
                    rf = list(model.random_forest_classifier())

                    classification_funcs = {
                        "SVM": s[0],
                        "Logistic Regression": lor[0],
                        "Decision Tree": dt[0],
                        "KNN": knn[0],
                        "Naive Bayes": nb[0],
                        "AdaBoost": ab[0],
                        "Multi-Layer Perceptron": mlp[0],
                        "Gradient Boosting": gb[0],
                        "Random Forest": rf[0]
                    }

                    metrics = list(s[1].keys())

                    classifiers_table = {
                        "SVM": list(s[1].values()),
                        "Logistic Regression": list(lor[1].values()),
                        "Decision Tree": list(dt[1].values()),
                        "KNN": list(knn[1].values()),
                        "Naive Bayes": list(nb[1].values()),
                        "AdaBoost": list(ab[1].values()),
                        "Multi-Layer Perceptron": list(mlp[1].values()),
                        "Gradient Boosting": list(gb[1].values()),
                        "Random Forest": list(rf[1].values())
                    }
                    comp_table = pd.DataFrame.from_dict(classifiers_table)
                    comp_table.index = metrics
                    comp_table = comp_table.transpose()
                
                elif state == 3:
                    clustering_models = ["KMeans", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"]
                    km = list(model.kmeans())
                    ac = list(model.agglomerative_clustering())
                    db = list(model.dbscan())
                    gm = list(model.gaussian_mixture())

                    clustering_funcs = {
                        "KMeans": km[0],
                        "Agglomerative Clustering": ac[0],
                        "DBSCAN": db[0],
                        "Gaussian Mixture": gm[0]
                    }

                    metrics = list(km[1].keys())

                    clustering_table = {
                        "KMeans": list(km[1].values()),
                        "Agglomerative Clustering": list(ac[1].values()),
                        "DBSCAN": list(db[1].values()),
                        "Gaussian Mixture": list(gm[1].values())
                    }
                    comp_table = pd.DataFrame.from_dict(clustering_table)
                    comp_table.index = metrics
                    comp_table = comp_table.transpose()

            if state != 4:
                if comp_table_flag == 1:
                    st.subheader("Model Comparison Table:")
                    st.dataframe(comp_table)

                    if state == 1:
                        st.subheader("Select Regressor for Prediction:")
                        regressor_selected = st.selectbox("Regressors", regression_models)
                        regressor = regression_funcs[regressor_selected]
                        user_input = [float(x) for x in st.text_input("Enter input features separated by commas:").split(",")]
                        if st.button("Predict"):
                            prediction = regressor(user_input)
                            st.write("Prediction:", prediction)

                    elif state == 2:
                        st.subheader("Select Classifier for Prediction:")
                        classifier_selected = st.selectbox("Classifiers", classification_models)
                        classifier = classification_funcs[classifier_selected]
                        user_input = [float(x) for x in st.text_input("Enter input features separated by commas:").split(",")]
                        if st.button("Predict"):
                            prediction = classifier(user_input)
                            st.write("Prediction:", prediction)

                    elif state == 3:
                        st.subheader("Select Clustering Algorithm:")
                        clustering_selected = st.selectbox("Clustering Algorithms", clustering_models)
                        clustering = clustering_funcs[clustering_selected]
                        user_input = [float(x) for x in st.text_input("Enter input features separated by commas:").split(",")]
                        if st.button("Cluster"):
                            cluster = clustering(user_input)
                            st.write("Cluster:", cluster)

if __name__ == "__main__":
    if "username" not in st.session_state:
        login_or_signup = st.sidebar.selectbox("Login or Sign Up", ["Login", "Sign Up"])
        if login_or_signup == "Login":
            login()
        elif login_or_signup == "Sign Up":
            signup()
    else:
        main()
