import tkinter as tk
from tkinter import ttk, messagebox, Canvas
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def create_step_button(frame, number, text, button_width):
    # Create the label for the number
    number_label = tk.Label(
        frame, text=number, font=("Montserrat Black", 24, "bold"), bg="#FFAE00", fg="black"
    )
    number_label.pack(side="left", padx=10)

    # Create the button for the step
    button = tk.Button(
        frame,
        text=text,
        font=("Montserrat", 16, "bold"),
        bg="black",
        fg="#FFAE00",
        relief="flat",
        width=button_width,
        height=2,
    )
    button.pack(side="left", padx=20)
    return button


class MedicalInsuranceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Insurance Prediction")
        self.root.geometry("1920x1080")
        self.root.configure(bg="#FFAE00")

        # Create a Scrollable Frame
        self.canvas = Canvas(root, bg="#FFAE00")
        self.scrollable_frame = tk.Frame(self.canvas, bg="#FFAE00")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Initialize variables
        self.df = None
        self.model = None
        self.preprocessor = None
        self.viz_window = None

        # Fonts
        self.header_font = ("Montserrat Black", 36, "bold")
        self.subheader_font = ("Montserrat", 18, "bold")
        self.label_font = ("Montserrat", 18, "bold")
        self.button_font = ("Montserrat", 16, "bold")
        self.entry_font = ("Montserrat", 14)
        # Create UI
        self.create_ui()

    def create_ui(self):
        # Title
        title = tk.Label(
            self.scrollable_frame,
            text="MEDICAL INSURANCE PREDICTION",
            font=self.header_font,
            bg="#FFAE00",
            fg="black",
        )
        title.pack(pady=30)

        # Subtitle
        subtitle = tk.Label(
            self.scrollable_frame,
            text="A modern-day program designed to predict insurance costs using a variety of everyday factors\nand implementing it using an up-to-date machine learning model!",
            font=self.subheader_font,
            bg="#FFAE00",
            fg="black",
            justify="center",
        )
        subtitle.pack(pady=20)

        # Steps section
        steps_subtitle = tk.Label(
            self.scrollable_frame,
            text="FOLLOW THESE STEPS",
            font=self.subheader_font,
            bg="#FFAE00",
            fg="black",
        )
        steps_subtitle.pack(pady=20)

        # Steps frame
        steps_frame = tk.Frame(self.scrollable_frame, bg="#FFAE00")
        steps_frame.pack(pady=20)

        # Create Step Buttons with functionality
        step1_frame = tk.Frame(steps_frame, bg="#FFAE00")
        step1_frame.grid(row=0, column=0, padx=30)
        load_button = create_step_button(step1_frame, "01", "LOAD INSURANCE DATASET", 25)
        load_button.config(command=self.load_insurance_data)

        step2_frame = tk.Frame(steps_frame, bg="#FFAE00")
        step2_frame.grid(row=0, column=1, padx=30)
        train_button = create_step_button(step2_frame, "02", "TRAIN MODEL", 25)
        train_button.config(command=self.train_model)

        step3_frame = tk.Frame(steps_frame, bg="#FFAE00")
        step3_frame.grid(row=0, column=2, padx=30)
        viz_button = create_step_button(step3_frame, "03", "DATA VISUALISATIONS", 25)
        viz_button.config(command=self.open_visualizations)

        # User details section
        details_label = tk.Label(
            self.scrollable_frame,
            text="ENTER YOUR DETAILS HERE",
            font=self.label_font,
            bg="#FFAE00",
            fg="black",
        )
        details_label.pack(pady=40)

        # Input fields frame
        inputs_frame = tk.Frame(self.scrollable_frame, bg="#FFAE00")
        inputs_frame.pack(pady=10)

        # Input entries dictionary
        self.input_entries = {}

        # First Row: CHILDREN, REGION, SMOKER
        self.create_input_field(inputs_frame, "CHILDREN", 0, 0, is_dropdown=False)
        self.create_input_field(inputs_frame, "REGION", 0, 2,
                                dropdown_values=["Southwest", "Southeast", "Northwest", "Northeast"], is_dropdown=True)
        self.create_input_field(inputs_frame, "SMOKER", 0, 4,
                                dropdown_values=["Yes", "No"], is_dropdown=True)

        # Second Row: AGE, BMI, SEX
        self.create_input_field(inputs_frame, "AGE", 1, 0, is_dropdown=False)
        self.create_input_field(inputs_frame, "BMI", 1, 2, is_dropdown=False)
        self.create_input_field(inputs_frame, "SEX", 1, 4,
                                dropdown_values=["Male", "Female"], is_dropdown=True)

        # Predict Button
        predict_button = tk.Button(
            self.scrollable_frame,
            text="PREDICT INSURANCE COST",
            font=self.button_font,
            bg="black",
            fg="#FFAE00",
            relief="flat",
            width=30,
            height=2,
            command=self.predict_cost
        )
        predict_button.pack(pady=40)

        # Result Frame
        self.result_frame = tk.Frame(self.scrollable_frame, bg="black", width=1000, height=200)
        self.result_frame.pack(pady=20)

        # Result Label
        self.result_label = tk.Label(
            self.result_frame,
            text="Details and Predicted Cost will be displayed here...",
            font=("Montserrat", 14),
            bg="black",
            fg="#FFAE00",
            wraplength=900,
            justify="center",
        )
        self.result_label.place(relx=0.5, rely=0.5, anchor="center")

        # Footer
        footer_frame = tk.Frame(self.scrollable_frame, bg="#FFAE00")
        footer_frame.pack(side="bottom", fill="x")

        footer = tk.Label(
            footer_frame,
            text="A Machine Learning Project By: Syed Ukkashah Ahmed | Arsh Al Aman | Ibrahim Johar",
            font=("Montserrat", 12, "bold"),
            bg="#FFAE00",
            fg="black",
            anchor="center"
        )
        footer.pack(fill="both", expand=True)

    def create_input_field(self, parent_frame, label_text, row, column,
                           is_dropdown=False, dropdown_values=None):
        # Create label
        label = tk.Label(
            parent_frame,
            text=label_text,
            font=self.label_font,
            bg="#FFAE00",
            fg="black"
        )
        label.grid(row=row, column=column, padx=20, pady=10, sticky="e")

        # Create input field
        if is_dropdown:
            input_var = tk.StringVar()
            input_field = ttk.Combobox(
                parent_frame,
                textvariable=input_var,
                font=self.entry_font,
                state="readonly",
                width=15
            )
            input_field['values'] = dropdown_values
        else:
            input_var = tk.StringVar()
            input_field = tk.Entry(
                parent_frame,
                textvariable=input_var,
                font=self.entry_font,
                width=15,
                bg="black",
                fg="white"
            )

        input_field.grid(row=row, column=column + 1, padx=10, pady=10)
        self.input_entries[label_text.lower()] = input_var
        return input_field

    def load_insurance_data(self):
        try:
            # Use a predefined path
            file_path = 'insurance.csv'
            self.df = pd.read_csv(file_path)

            # Clear previous content in the result frame
            self.clear_result_frame()

            # Format dataset as a table using `tabulate`
            tabulated_data = tabulate(self.df.head(20), headers='keys', tablefmt='grid', showindex=False)

            # Create a Text widget to display the tabulated data
            text_widget = tk.Text(self.result_frame, wrap="none", bg="black", fg="#FFAE00", font=("Courier", 12))
            text_widget.pack(expand=True, fill="both", padx=10, pady=10)

            # Insert the tabulated data into the Text widget
            text_widget.insert("1.0", tabulated_data)

            # Disable editing of the text widget
            text_widget.configure(state="disabled")

            # Notify the user of successful data loading
            messagebox.showinfo("Data Loaded", "Insurance dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def clear_result_frame(self):
        # Clear all widgets in the result frame
        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def train_model(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load the dataset first!")
            return

        try:
            # Preprocessing
            X = self.df.drop(columns=["charges"])
            y = self.df["charges"]

            # Define categorical and numerical columns
            categorical_cols = ["sex", "smoker", "region"]
            numerical_cols = ["age", "bmi", "children"]

            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop='first'), categorical_cols)
                ]
            )

            # Transform the data
            X_transformed = preprocessor.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save the model and preprocessor
            self.model = model
            self.preprocessor = preprocessor

            # Clear previous content in the result frame
            self.clear_result_frame()

            # Update result label
            result_text = (
                f"Model Training Complete!\n"
                f"Mean Squared Error: {mse:.2f}\n"
                f"R² Score: {r2:.2f}"
            )
            self.result_label = tk.Label(
                self.result_frame,
                text=result_text,
                font=("Montserrat", 14),
                bg="black",
                fg="#FFAE00",
                wraplength=900,
                justify="center",
            )
            self.result_label.place(relx=0.5, rely=0.5, anchor="center")

            messagebox.showinfo("Training Complete", "Model trained successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Model training failed: {str(e)}")

    def predict_cost(self):
        # Check if model is trained
        if self.model is None or self.preprocessor is None:
            messagebox.showerror("Error", "Please train the model first!")
            return

        try:
            # Collect input values
            input_data = {
                'age': float(self.input_entries['age'].get()),
                'bmi': float(self.input_entries['bmi'].get()),
                'children': int(self.input_entries['children'].get()),
                'sex': self.input_entries['sex'].get().lower(),
                'smoker': self.input_entries['smoker'].get().lower(),
                'region': self.input_entries['region'].get().lower()
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Preprocess the input data
            input_transformed = self.preprocessor.transform(input_df)

            # Make prediction
            prediction = self.model.predict(input_transformed)[0]

            # Clear previous content in the result frame
            self.clear_result_frame()

            # Display result
            result_text = (
                f"Predicted Insurance Cost: ${prediction:,.2f}\n\n"
                f"Input Details:\n"
                f"Age: {input_data['age']}\n"
                f"BMI: {input_data['bmi']}\n"
                f"Children: {input_data['children']}\n"
                f"Sex: {input_data['sex']}\n"
                f"Smoker: {input_data['smoker']}\n"
                f"Region: {input_data['region']}"
            )
            self.result_label = tk.Label(
                self.result_frame,
                text=result_text,
                font=("Montserrat", 14),
                bg="black",
                fg="#FFAE00",
                wraplength=900,
                justify="center",
            )
            self.result_label.place(relx=0.5, rely=0.5, anchor="center")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def open_visualizations(self):
        # Check if dataset is loaded
        if self.df is None:
            messagebox.showerror("Error", "Please load the dataset first!")
            return

        # Create visualization window
        if self.viz_window is None or not tk.Toplevel.winfo_exists(self.viz_window):
            self.viz_window = tk.Toplevel(self.root)
            self.viz_window.title("Data Visualizations")
            self.viz_window.geometry("1200x900")
            self.viz_window.configure(bg="#FFAE00")

            # Create a notebook (tabbed interface)
            notebook = ttk.Notebook(self.viz_window)
            notebook.pack(expand=True, fill='both', padx=10, pady=10)

            # Correlation Heatmap Tab
            heatmap_frame = tk.Frame(notebook, bg="#FFAE00")
            notebook.add(heatmap_frame, text="Correlation Heatmap")
            self.create_correlation_heatmap(heatmap_frame)

            # Scatter Plot (Age vs BMI vs Charges) Tab
            scatter_frame = tk.Frame(notebook, bg="#FFAE00")
            notebook.add(scatter_frame, text="Age-BMI-Charges Scatter")
            self.create_age_bmi_charges_scatter(scatter_frame)

            # Distribution Plot Tab
            dist_frame = tk.Frame(notebook, bg="#FFAE00")
            notebook.add(dist_frame, text="Charges Distribution")
            self.create_charges_distribution(dist_frame)

            # BMI vs Charges Tab
            bmi_charges_frame = tk.Frame(notebook, bg="#FFAE00")
            notebook.add(bmi_charges_frame, text="BMI vs Charges")
            self.create_bmi_charges_plot(bmi_charges_frame)

            # Age vs Charges Tab
            age_charges_frame = tk.Frame(notebook, bg="#FFAE00")
            notebook.add(age_charges_frame, text="Age vs Charges")
            self.create_age_charges_plot(age_charges_frame)

            # Smoker vs Charges Tab
            smoker_charges_frame = tk.Frame(notebook, bg="#FFAE00")
            notebook.add(smoker_charges_frame, text="Smoker vs Charges")
            self.create_smoker_charges_plot(smoker_charges_frame)

            # Add protocol to handle window close
            def on_viz_window_close():
                self.viz_window.destroy()
                self.viz_window = None

            self.viz_window.protocol("WM_DELETE_WINDOW", on_viz_window_close)

        else:
            self.viz_window.lift()

    def create_bmi_charges_plot(self, parent_frame):
        # Create figure and canvas
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FFAE00")
        fig.patch.set_facecolor("#FFAE00")

        # Create scatter plot for BMI vs Charges
        sns.scatterplot(
            data=self.df,
            x='bmi',
            y='charges',
            hue='smoker',
            palette=['blue', 'red'],
            ax=ax
        )

        # Customize plot
        ax.set_title("BMI vs Insurance Charges", fontsize=15, color='black')
        ax.set_xlabel("BMI", fontsize=12, color='black')
        ax.set_ylabel("Insurance Charges", fontsize=12, color='black')

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill='both', padx=10, pady=10)

    def create_age_charges_plot(self, parent_frame):
        # Create figure and canvas
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FFAE00")
        fig.patch.set_facecolor("#FFAE00")

        # Create scatter plot for Age vs Charges
        sns.scatterplot(
            data=self.df,
            x='age',
            y='charges',
            hue='smoker',
            palette=['blue', 'red'],
            ax=ax
        )

        # Customize plot
        ax.set_title("Age vs Insurance Charges", fontsize=15, color='black')
        ax.set_xlabel("Age", fontsize=12, color='black')
        ax.set_ylabel("Insurance Charges", fontsize=12, color='black')

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill='both', padx=10, pady=10)

    def create_smoker_charges_plot(self, parent_frame):
        # Create figure and canvas
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FFAE00")
        fig.patch.set_facecolor("#FFAE00")

        # Create box plot for Smoker vs Charges
        sns.boxplot(
            data=self.df,
            x='smoker',
            y='charges',
            palette=['blue', 'red'],
            ax=ax
        )

        # Customize plot
        ax.set_title("Smoker vs Insurance Charges", fontsize=15, color='black')
        ax.set_xlabel("Smoker", fontsize=12, color='black')
        ax.set_ylabel("Insurance Charges", fontsize=12, color='black')

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill='both', padx=10, pady=10)

    def create_correlation_heatmap(self, parent_frame):
        # Select numerical columns
        numerical_cols = ['age', 'bmi', 'children', 'charges']

        # Create figure and canvas
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FFAE00")
        fig.patch.set_facecolor("#FFAE00")

        # Compute correlation matrix
        corr_matrix = self.df[numerical_cols].corr()

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                    cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Correlation Heatmap of Numerical Features', fontsize=15, color='black')

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True, fill='both', padx=10, pady=10)

    def create_age_bmi_charges_scatter(self, parent_frame):
            # Create figure and canvas
            fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FFAE00")
            fig.patch.set_facecolor("#FFAE00")

            # Create scatter plot
            scatter = ax.scatter(
                self.df['age'],
                self.df['bmi'],
                c=self.df['charges'],
                cmap='viridis',
                alpha=0.7
            )

            # Customize plot
            ax.set_xlabel('Age', fontsize=12, color='black')
            ax.set_ylabel('BMI', fontsize=12, color='black')
            ax.set_title('Age vs BMI Colored by Insurance Charges', fontsize=15, color='black')

            # Add colorbar
            plt.colorbar(scatter, label='Insurance Charges')

            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill='both', padx=10, pady=10)

    def create_charges_distribution(self, parent_frame):
            # Create figure and canvas
            fig, ax = plt.subplots(figsize=(10, 8), facecolor="#FFAE00")
            fig.patch.set_facecolor("#FFAE00")

            # Create distribution plot
            sns.histplot(
                self.df['charges'],
                kde=True,
                color='#FFAE00',
                ax=ax
            )

            # Customize plot
            ax.set_xlabel('Insurance Charges', fontsize=12, color='black')
            ax.set_ylabel('Frequency', fontsize=12, color='black')
            ax.set_title('Distribution of Insurance Charges', fontsize=15, color='black')

            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=parent_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(expand=True, fill='both', padx=10, pady=10)


# Add this at the end of the script
if __name__ == "__main__":
    root = tk.Tk()
    app = MedicalInsuranceApp(root)
    root.mainloop()