import sys
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from keras.src.datasets.boston_housing import load_data
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.impute import SimpleImputer
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None

        # Neural network configuration
        self.layer_config = []

        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()

            if dataset_name == "Load Custom Dataset":
                return

            # Load selected dataset
            if dataset_name == "Iris Dataset":
                self.data = datasets.load_iris()

            elif dataset_name == "Breast Cancer Dataset":
                self.data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                self.data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                self.data = fetch_california_housing()

            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return

            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(self.data.data, self.data.target,
                                                 test_size=test_size,
                                                 random_state=42)

            print("oy")
            # İç içe listelerdeki kayıp verileri (None) kontrol et
            missing_values = [[item is None for item in inner_list] for inner_list in self.X_train]

            print(missing_values)

            # Apply scaling if selected
            self.apply_scaling()

            self.status_bar.showMessage(f"Loaded {dataset_name}")

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )

            if file_name:
                # Load data
                data = pd.read_csv(file_name)

                # Ask user to select target column
                target_col = self.select_target_column(data.columns)

                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]

                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y,
                                                         test_size=test_size,
                                                         random_state=42)

                    # Apply scaling if selected
                    self.apply_scaling()

                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")

        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)

        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()

        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()

                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)

            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()

        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])

        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)

        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()

        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()


        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox",
             }
        )
        self.lr_loss_combo = QComboBox()
        self.lr_loss_combo.addItems(["MSE","MAE"])
        self.lr_loss_combo.currentIndexChanged.connect(self.on_loss_func_change)
        lr_group.layout().addWidget(self.lr_loss_combo)  # # Add it inside Linear Regression group

        regression_layout.addWidget(lr_group)

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0)
        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double"}
        )
        classification_layout.addWidget(nb_group)

        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)

        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)

        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)

        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)

        return widget

    def on_loss_func_change(self):
        """Handle the change in the loss function selection"""
        selected_loss_func = self.lr_loss_combo.currentText()
        print("dede")

        # Print or perform operations based on the selected loss function
        if selected_loss_func == "MAE":
            print("MAE selected, perform MAE related tasks.")
        elif selected_loss_func == "MSE":
            print("MSE selected, perform MSE related tasks.")

    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()

        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)

        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)

        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()

        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)

        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)

        return widget

    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()

        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        #sil
        self.env_combo.currentIndexChanged.connect(self.on_env_change)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)

        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()

        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)

        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)

        return widget

    def on_env_change(self):
        """Çevre değiştiğinde çağrılan fonksiyon"""
        selected_env = self.env_combo.currentText()

        if selected_env == "CartPole-v1":
            print("hey")
        elif selected_env == "MountainCar-v0":
            print("hoi")
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.figure2 = Figure(figsize=(8, 6))  # Yeni figure
        self.canvas2 = FigureCanvas(self.figure2)  # Yeni canvas
        viz_layout.addWidget(self.canvas2)  #
        viz_layout.addWidget(self.canvas)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    def plot_graph_predict(self,x_dataset):
        data = x_dataset['data']
        print("hey")
        print(self.y_pred_house)
        target = x_dataset['target']
        feature_names = x_dataset['feature_names']

        # California Housing Dataset için bir check
        if 'MedInc' in feature_names:
            print("hey")
            # Pandas DataFrame'e dönüştür

            print("heyo")
            # Eski grafiği temizle
            self.figure.clear()

            # Yeni bir eksen oluştur
            ax = self.figure.add_subplot(111)

            print(len(self.y_pred_house))
            print(len(self.X_test[:,0]))
            # Medyan Gelir (MedInc) ve Gerçek Ev Fiyatı ilişkisini çiz
            ax.scatter(self.X_test[:,0], self.y_test, color="blue", alpha=0.5,
                       label="Gerçek Ev Fiyatları")

            # Medyan Gelir (MedInc) ve Tahmin Edilen Ev Fiyatı ilişkisini kırmızı çarpılarla çiz
            ax.scatter(self.X_test[:,0], self.y_pred_house, color="red", marker='x', alpha=0.5,
                       label="Tahmin Edilen Ev Fiyatları")
            print("hey")
            # Grafik başlıkları ve etiketler
            ax.set_title('Medyan Gelire Göre Ev Fiyatları')
            ax.set_xlabel('Medyan Gelir (MedInc)')
            ax.set_ylabel('Ev Fiyatı')

            # Legendi ekle
            ax.legend()

            # Çizimi güncelle ve canvas'a aktar
            self.canvas.draw()
        if feature_names[0] == 'sepal length (cm)':
            self.figure.clear()

            # Yeni bir eksen oluştur
            ax = self.figure.add_subplot(122)

            print("Iris Veri Seti Tahmin Grafiği")
            print("Gerçek Etiket Sayısı:", len(self.y_test))
            print("Tahmin Sayısı:", len(self.y_pred_iris))

            # Gerçek sınıfları mavi noktalar olarak çiz
            ax.scatter(self.X_test[:, 0], self.y_test, color="blue", alpha=0.5,
                       label="Gerçek Sınıflar")

            # Tahmin edilen sınıfları kırmızı çarpılar olarak çiz
            ax.scatter(self.X_test[:, 0], self.y_pred_iris, color="red", marker='x', alpha=0.5,
                       label="Tahmin Edilen Sınıflar")

            # Grafik başlıkları ve etiketler
            ax.set_title('Logistic Regression - Iris Dataset')
            ax.set_xlabel('Sepal Length')
            ax.set_ylabel('Sınıf')

            # Legendi ekle
            ax.legend()

            # Çizimi güncelle ve canvas'a aktar
            self.canvas.draw()

    def plot_graph_raw(self,x_dataset):

        data = x_dataset['data']
        target = x_dataset['target']
        feature_names = x_dataset['feature_names']
        # if feature_names[0] == 'sepal length (cm)':
        #     # Eğer iris dataseti seçilirse bu şekilde görselleştirilecek raw data
        #     # Pandas DataFrame'e dönüştür
        #     x_dataset_df = pd.DataFrame(data, columns=feature_names)
        #     x_dataset_df['Species'] = target
        #
        #     # Önce eski grafiği temizle
        #     self.figure2.clear()
        #
        #     # Yeni bir eksen oluştur
        #     ax = self.figure2.add_subplot(111)
        #
        #     # Her bir tür için scatter plot çiz
        #     for species in x_dataset_df['Species'].unique():
        #         subset = x_dataset_df[x_dataset_df['Species'] == species]
        #         ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=f"Species {species}")
        #
        #     # Grafik başlıkları ve etiketler
        #     ax.set_title('Sepal Length vs Sepal Width')
        #     ax.set_xlabel('Sepal Length')
        #     ax.set_ylabel('Sepal Width')
        #
        #     # Legendi ekle
        #     ax.legend()
        #
        #     # Çizimi güncelle ve canvas'a aktar
        #     self.canvas2.draw()
        if 'MedInc' in feature_names:
            # Pandas DataFrame'e dönüştür
            x_dataset_df = pd.DataFrame(data, columns=feature_names)
            x_dataset_df['House Price'] = target

            # Eski grafiği temizle
            self.figure2.clear()

            # Yeni bir eksen oluştur
            ax = self.figure2.add_subplot(111)

            # Medyan Gelir (MedInc) ve Ev Fiyatı ilişkisini çiz
            ax.scatter(x_dataset_df['MedInc'], x_dataset_df['House Price'], color="blue", alpha=0.5,
                       label="Ev Fiyatları")

            # Grafik başlıkları ve etiketler
            ax.set_title('Medyan Gelire Göre Ev Fiyatları')
            ax.set_xlabel('Medyan Gelir (MedInc)')
            ax.set_ylabel('Ev Fiyatı')

            # Legendi ekle
            ax.legend()

            # Çizimi güncelle ve canvas'a aktar
            self.canvas2.draw()


    def train_linear_regression_model(self, fit_intercept=True, normalize=False):
        X = self.X_train
        y = self.y_train

        scaler = None
        if normalize:
            scaler = preprocessing.StandardScaler()
            X = scaler.fit_transform(X)

        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X, y)
        print("hola")
        self.y_pred_house = model.predict(self.X_test)
        print(len(self.X_test))
        print(len(self.y_test))
        print("Test seti tahminleri:", self.y_pred_house)
        # Buraya daha param seçimi yapılacak unutulmasın.

    """def predict_linear_regression(self, scaler=None):
        X_test = self.X_train
        if scaler:
            X_test = scaler.transform(X_test)
        return model.predict(X_test)"""

    def logistic_regression(self, C=1.0, max_iter=100, multi_class='ovr'):
        """
        Logistic Regression modelini verilen parametrelerle eğitir.

        Args:
        - C (float): Regularization strength. Daha düşük değerler daha fazla regularizasyon.
        - max_iter (int): Maksimum iterasyon sayısı.
        - multi_class (str): 'ovr' (one-vs-rest) ya da 'multinomial' (multinomial loss) sınıflandırma stratejisi.
        - X_train (array-like): Eğitim verisi (özellikler).
        - y_train (array-like): Eğitim verisi (etiketler).

        Returns:
        - model (LogisticRegression): Eğitilmiş Logistic Regression modelini döndürür.
        """
        # Modeli oluştur
        #C = self.param_widgets["C"].value()

        model = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class)

        # Modeli eğit
        model.fit(self.X_train, self.y_train)

        print(f"Logistic Regression model trained with C={C}, max_iter={max_iter}, multi_class={multi_class}")
        self.y_pred_iris = model.predict(self.X_test)
        print(len(self.y_pred_iris))

    # Kullanım örneği
    # X_train, y_train veri setinizin özellikler ve etiketler kısmı olmalı
    # model = logistic_regression(C=1.0, max_iter=200, multi_class='ovr', X_train=X_train, y_train=y_train)
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()

        # Create parameter inputs
        self.param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)

            param_layout.addWidget(widget)
            self.param_widgets[param_name] = widget
            layout.addLayout(param_layout)
            # param_widgets ben ekledim self ifadesini bişiler deniyorussss. Tez vakitte düzeltile.

        # Add train button
        train_btn = QPushButton(f"Train {name}")
        if name == "Linear Regression":
            train_btn.clicked.connect(lambda: [self.plot_graph_raw(self.data), self.train_linear_regression_model(),
                                               self.plot_graph_predict(self.data),"""self.wression()"""])
            layout.addWidget(train_btn)
        if name == "Logistic Regression":
            train_btn.clicked.connect(lambda: [self.plot_graph_raw(self.data), self.train_linear_regression_model(),
                                               self.plot_graph_predict(self.data)])
            layout.addWidget(train_btn)
        if name == "Naive Bayes":
            train_btn.clicked.connect(lambda: [self.plot_graph_raw(self.data), self.train_linear_regression_model(),
                                               self.plot_graph_predict(self.data)])
            layout.addWidget(train_btn)
        """train_btn.clicked.connect(lambda: [self.plot_graph_raw(self.data),self.train_linear_regression_model(), self.plot_graph_predict(self.data)])
        layout.addWidget(train_btn)"""
        # Şunu açarsam bütün tuşlar gözükür ama benim amacım sadece basılı olan tuş fonksiyon çağırsın. O yüzden böylesi daha iyi.
        group.setLayout(layout)
        return group

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()

        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)

        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)

        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)

        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)

        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()

        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)

        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)

        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()

        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)

        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)

        return widget

    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)

        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()

        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}

        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input

                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)

            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input

                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input

                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)

            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)

        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def add_layer():
            layer_type = type_combo.currentText()

            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))

            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })

            dialog.accept()

        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)

        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        group.setLayout(layout)
        return group

    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()

        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()

        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)

        group.setLayout(layout)
        return group

    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return

        try:
            # Create and compile model
            model = self.create_neural_network()

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()

            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test

            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)

            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])

            # Update visualization with training history
            self.plot_training_history(history)

            self.status_bar.showMessage("Neural Network Training Complete")

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()

        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]

            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))

        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model

    def train_neural_network(self):
        """Train the neural network"""
        try:
            # Create and compile model
            model = self.create_neural_network()

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()

            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Train model
            history = model.fit(self.X_train, self.y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(self.X_test, self.y_test),
                                callbacks=[self.create_progress_callback()])

            # Update visualization with training history
            self.plot_training_history(history)

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")

    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""

        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar

            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)

        return ProgressCallback(self.progress_bar)

    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()

        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")

        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)

                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                     c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)

            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                     c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)

        self.canvas.draw()

    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"

        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)

            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"

        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)

        self.metrics_text.setText(metrics_text)

    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()

        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])

        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])

        self.figure.tight_layout()
        self.canvas.draw()

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)


def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

