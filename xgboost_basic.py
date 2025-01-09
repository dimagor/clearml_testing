import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from clearml import Task
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import plot_tree


def train(params):

    df_credit = pd.read_csv("data.csv", index_col=0)

    # Fill in the missing values in the fields below
    df_credit["Saving accounts"] = df_credit["Saving accounts"].fillna("no_inf")
    df_credit["Checking account"] = df_credit["Checking account"].fillna("no_inf")

    # Convert Sex to binary category
    df_credit["Sex"] = df_credit["Sex"].map({"male": 1, "female": 0})

    # Convert Saving accounts, Checking account, Purpose, and Housing into "category" types
    df_credit["Saving accounts"] = df_credit["Saving accounts"].astype("category")
    df_credit["Checking account"] = df_credit["Checking account"].astype("category")
    df_credit["Purpose"] = df_credit["Purpose"].astype("category")
    df_credit["Housing"] = df_credit["Housing"].astype("category")

    # Convert Risk into a binary if Risk is bad
    df_credit["Risk_bad"] = df_credit["Risk"].apply(lambda x: 1 if x == "bad" else 0)
    del df_credit["Risk"]

    # task.register_artifact("train", df_credit)

    # Creating the X and y variables
    X = df_credit.drop(["Risk_bad"], axis=1)
    y = df_credit["Risk_bad"]

    # Splitting X and y into train and test version
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Debug test
    print(X_train.head(5))

    # Load the data into XGBoost format
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # Train the XGBoost Model
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=25,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=0,
    )

    # Save the model
    bst.save_model("best_model")

    # Make predictions for test data
    y_pred = bst.predict(dtest)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Plots
    # plot_tree(bst)
    # plt.title("Decision Tree")
    # plt.show()


if __name__ == "__main__":
    # Set the parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["error", "auc"],
        "max_depth": 3,  # the maximum depth of each tree
        "eta": 0.3,  # the training step for each iteration
        "gamma": 0,
        "max_delta_step": 1,
        "subsample": 1,
        "sampling_method": "uniform",
        "seed": 1337,
    }
    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(
        project_name="Basic Testing",
        task_name="XGBoost simple example",
        output_uri=False,  # NOTE: This causes everything to hang!
    )
    task.connect(params)
    train(params)
    print("Done!")
