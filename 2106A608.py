import sys
import numpy as np
import pandas as pd
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,QInputDialog,
                             QProgressBar, QCheckBox,QListWidget, QGridLayout, QMessageBox,
                             QDialog, QLineEdit)
from sklearn.preprocessing import StandardScaler
from PyQt6.QtCore import Qt, pyqtSlot
import matplotlib.pyplot as plt
from fastai.metrics import perplexity
from fastai.vision.models.xresnet import xse_resnext18
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from PyQt6.QtWidgets import QFileDialog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.datasets import fetch_california_housing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE



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
        self.data = None
        self.X_train = None
        self.X_cv = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None



        # Neural network configuration
        self.layer_config = []
        self.cnn_layers_list = None
        self.selected_optimizer = None
        self.model = None
        self.rnn_layer_config = []
        self.rnn_layers_list = None


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
            elif dataset_name == "IMDB":
                num_words = 5000
                (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)



            elif dataset_name == "Digits Dataset":
                self.data = datasets.load_digits()
            elif dataset_name == "California Housing Dataset":
                self.data = datasets.fetch_california_housing()

            elif dataset_name == "MNIST Dataset":
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


                # Verileri birleştir
                X_all = np.concatenate([x_train, x_test], axis=0)

                y_all = np.concatenate([y_train, y_test], axis=0)

                # Flatten (28x28 → 784)

                # CNN'ler için 4D şekil: (num_samples, height, width, channels)
                X_all = X_all.reshape(-1, 28, 28, 1)  # 1 = grayscale channel

                # Artık elinde tek parça veri var, istediğin gibi bölebilirsin:
                self.data = {'data': X_all, 'target': y_all}

            # default olarak k fold seçili dursun madem
            #self.split_dataset() Bu kısım sadece raw veririy görselltirşem için
            if dataset_name != "MNIST Dataset" and dataset_name != "IMDB":

                self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                    self.data.data, self.data.target, test_size=0.2, random_state=42)

            else:
                if dataset_name == "MNIST Dataset":
                    self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
                        self.data['data'], self.data['target'], test_size=0.2, random_state=42)

                    self.y_train = to_categorical(self.y_train, num_classes=10)
                    self.y_test = to_categorical(self.y_test, num_classes=10)
                if dataset_name == "IMDB":
                    X_train = pad_sequences(X_train, maxlen=200)
                    X_test = pad_sequences(X_test, maxlen=200)
                    self.X_train = pad_sequences(X_train, maxlen=200)
                    self.X_test = pad_sequences(X_test, maxlen=200)


                    self.y_train = to_categorical(y_train, num_classes=2)
                    self.y_test = to_categorical(y_test, num_classes=2)
                    print(y_train.shape) # (25000,2)   0,1

            # Apply scaling if selected
            self.apply_scaling()




            self.status_bar.showMessage(f"Loaded {dataset_name}")

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def split_dataset(self):
        # Split data
        # Buraya bir try block gelecek veri seçilmeden girilmesin diye.

        selected_method = self.method_combo.currentText()

        if selected_method == "Custom Split (Train/CV/Test)":
            # Normal hold-out veya custom oranlarla ayır
            # Kullanıcının girdiği oranlar
            train_ratio = self.train_ratio_input.value()  # Örneğin: 0.80
            cv_ratio = self.cv_ratio_input.value()  # Örneğin: 0.15
            test_ratio = self.test_ratio_input.value()  # Örneğin: 0.05



            # 1. Önce Train+CV ve Test olarak böl
            X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                self.data['data'],self.data['target'],
                test_size=test_ratio,
                random_state=42
            )

            # 2. Sonra Train ve CV olarak böl
            # Burada dikkat: kalan (X_temp) üzerinde bölme yapıyoruz
            cv_adjusted_ratio = cv_ratio / (train_ratio + cv_ratio)  # Oranı yeniden normalize ediyoruz

            self.X_train, self.X_cv, self.y_train, self.y_cv = model_selection.train_test_split(
                X_temp, y_temp,
                test_size=cv_adjusted_ratio,
                random_state=42
            )




        elif selected_method == "K-Fold Cross Validation":
            # K-Fold için split YOK! Çünkü her fold sırasında bölünecek
            self.k = self.k_spinbox.value()
            self.kfold = model_selection.KFold(n_splits=self.k, shuffle=True, random_state=42)
            # Model eğitim aşamasında, her fold için ayrı eğitim yapılacak.
            # Burada sadece hazır olmasını sağlıyoruz.


        else:
            self.X_train = self.data.data
            self.y_train = self.data.target


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

                # K-Fold yapılmadıysa, CV ve Test kümeleri scale edilmeli
                selected_method = self.method_combo.currentText()

                if selected_method == "Custom Split (Train/CV/Test)":
                    self.X_train = scaler.fit_transform(self.X_train)
                    self.X_cv = scaler.transform(self.X_cv)
                    self.X_test = scaler.transform(self.X_test)
                elif selected_method == "K-Fold Cross Validation":
                    # Sadece tüm veriyi scale et (test/cv yok çünkü)
                    self.X_train = scaler.fit_transform(self.X_train)
                else:
                    # Default/Diğer durumda sadece X_train ve X_test kullanılıyor gibi varsayıyoruz
                    self.X_train = scaler.fit_transform(self.X_train)
                    if hasattr(self, 'X_test'):
                        self.X_test = scaler.transform(self.X_test)

            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()  # DİKKAT: Artık dikey layout, çünkü altına başka grup ekleyeceğiz!

        # Dataset selection
        dataset_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "California Housing Dataset",
            "MNIST Dataset",
            "IMDB"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])

        dataset_layout.addWidget(QLabel("Dataset:"))
        dataset_layout.addWidget(self.dataset_combo)
        dataset_layout.addWidget(self.load_btn)
        dataset_layout.addWidget(QLabel("Scaling:"))
        dataset_layout.addWidget(self.scaling_combo)

        #Validation Settings Group
        validation_group = QGroupBox("Validation Settings")
        validation_layout = QHBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["K-Fold Cross Validation", "Custom Split (Train/CV/Test)"])

        self.method_combo.currentIndexChanged.connect(self.toggle_validation_options)

        self.k_label = QLabel("Select K (folds):")
        self.k_spinbox = QSpinBox()
        self.k_spinbox.setMinimum(2)
        self.k_spinbox.setMaximum(10)
        self.k_spinbox.setValue(5)

        self.train_ratio_label = QLabel("Train Ratio:")
        self.train_ratio_input = QDoubleSpinBox()
        self.train_ratio_input.setRange(0.1, 0.9)
        self.train_ratio_input.setSingleStep(0.05)
        self.train_ratio_input.setValue(0.7)

        self.cv_ratio_label = QLabel("CV Ratio:")
        self.cv_ratio_input = QDoubleSpinBox()
        self.cv_ratio_input.setRange(0.05, 0.8)
        self.cv_ratio_input.setSingleStep(0.05)
        self.cv_ratio_input.setValue(0.15)

        self.test_ratio_label = QLabel("Test Ratio:")
        self.test_ratio_input = QDoubleSpinBox()
        self.test_ratio_input.setRange(0.05, 0.8)
        self.test_ratio_input.setSingleStep(0.05)
        self.test_ratio_input.setValue(0.15)

        # Bunun sebebi en başta gizleme yapmak istemem.
        self.toggle_validation_options()

        # Layout içinde sırayla ekliyoruz
        validation_layout.addWidget(self.method_combo)

        kfold_layout = QHBoxLayout()
        kfold_layout.addWidget(self.k_label)
        kfold_layout.addWidget(self.k_spinbox)
        validation_layout.addLayout(kfold_layout)

        split_layout1 = QHBoxLayout()
        split_layout1.addWidget(self.train_ratio_label)
        split_layout1.addWidget(self.train_ratio_input)
        validation_layout.addLayout(split_layout1)

        split_layout2 = QHBoxLayout()
        split_layout2.addWidget(self.cv_ratio_label)
        split_layout2.addWidget(self.cv_ratio_input)
        validation_layout.addLayout(split_layout2)

        split_layout3 = QHBoxLayout()
        split_layout3.addWidget(self.test_ratio_label)
        split_layout3.addWidget(self.test_ratio_input)
        validation_layout.addLayout(split_layout3)

        self.ok_buton = QPushButton("Save Selections")
        split_layout4 = QHBoxLayout()
        split_layout4.addWidget(self.ok_buton)
        validation_layout.addLayout(split_layout4)
        self.ok_buton.clicked.connect(self.split_dataset)



        validation_group.setLayout(validation_layout)

        # --- Tüm grupları ana data layoutuna ekliyoruz ---
        data_layout.addLayout(dataset_layout)
        data_layout.addWidget(validation_group)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def toggle_validation_options(self):
        is_kfold = self.method_combo.currentText() == "K-Fold Cross Validation"
        self.k_label.setVisible(is_kfold)
        self.k_spinbox.setVisible(is_kfold)

        for widget in [self.train_ratio_label, self.train_ratio_input,
                       self.cv_ratio_label, self.cv_ratio_input,
                       self.test_ratio_label, self.test_ratio_input]:
            widget.setVisible(not is_kfold)

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
             "normalize": "checkbox"}
        )
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
    def elbow_method_graphic(self):
        try:
            k_values = range(1, 11)
            wcss = []
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
                kmeans.fit(self.X_train)
                wcss.append(kmeans.inertia_)
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(k_values, wcss, marker='o')
            ax.set_title('Elbow Method for Optimal k')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('WCSS (Inertia)')
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error you need to select data: {str(e)}")

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
             "with pca": "checkbox",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)

        self.centerBtn = QPushButton(text="Elbow Method Graphic", parent=self)
        self.centerBtn.clicked.connect(lambda: self.elbow_method_graphic())
        kmeans_layout.addWidget(self.centerBtn)

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


        lda_group = QGroupBox("Linear Discrimenet Analysis")
        lda_layout = QVBoxLayout()

        lda_params = self.create_algorithm_group(
            name="LDA Parameters",
            params={"n_components":"int"}
        )
        lda_layout.addWidget(lda_params)
        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 1, 1)

        tsne_group = QGroupBox("t-SNE Analysis(2D/3D)")
        tsne_layout = QVBoxLayout()
        tsne_params = self.create_algorithm_group(
            name="t-SNE Parameters",
            params={"n_components":"int",
                    "perplexity":"int",
                    "random_state":"int"
                    })
        tsne_layout.addWidget(tsne_params)
        tsne_group.setLayout(tsne_layout)
        layout.addWidget(tsne_group,1,0)

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

    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)

        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)

        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

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
        param_widgets = {}
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
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)

        # Add train button
        train_btn = QPushButton(f"Train {name}")

        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)

        group.setLayout(layout)
        return group
    def train_model(self,name,param_widgets):
        # Parametreleri oku
        params = {}

        for key, widget in param_widgets.items():
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentText()

        # Heee şimdi işte anladım nasıl widget çalıştığını ilgili train tuşuna basarsan şunun gibi bişi  gelecek
        """ooo
            oe
            Naive Bayes
            oe
            {'var_smoothing': <PyQt6.QtWidgets.QDoubleSpinBox object at 0x0000015463AF4670>}
            {'var_smoothing': 0.0}"""
        if name == "Linear Regression":

            fit_intercept = params.get("fit_intercept")

            normalize = params.get("normalize")

            self.linear_regression(normalize,fit_intercept)
        if name == "Logistic Regression":

            C = params.get("C")
            max_iter = params.get("max_iter")
            multi_class = params.get("multi_class")
            normalize = False
            self.logistic_regression(C, max_iter, multi_class, normalize)

        if name == "PCA Parameters":

            # PCA uygula ama model eğitme!
            pca = PCA(n_components=params.get("n_components", 2),
                      whiten=params.get("whiten", False))

            train_ratio = self.train_ratio_input.value()  # Örneğin: 0.80
            cv_ratio = self.cv_ratio_input.value()  # Örneğin: 0.15
            test_ratio = self.test_ratio_input.value()  # Örneğin: 0.05
            X_train_pca = pca.fit_transform(self.X_train)

            self.update_visualization(self.y_train, X_train_pca, params=params)

            explained_variance = pca.explained_variance_ratio_
            metrics_text = " The ratio of variance : \n"
            metrics_text += f"{explained_variance}"
            self.metrics_text.setText(metrics_text)

        if name == "LDA Parameters":
            try:
                # Parametreden n_components'ı oku
                n_comp = params.get("n_components", 2)
                n_classes = len(np.unique(self.y_train))

                # Hata kontrolü: LDA bileşen sayısı en fazla n_classes - 1 olabilir
                if n_comp > (n_classes - 1):
                    self.show_error(
                        f"n_components value too high. It could be at most {n_classes - 1}  "
                        f"(The current class number: {n_classes})."
                    )
                    return  # Fonksiyondan çık, LDA çalışmasın

                # LDA vakti geldi
                lda = LinearDiscriminantAnalysis(n_components=n_comp)

                # Eğitim verisine fit + transform
                X_train_lda = lda.fit_transform(self.X_train, self.y_train)

                # Görselleştirme için gerçek etiketleri kullan
                self.update_visualization(self.y_train, X_train_lda, params=params)
                explained_variance = lda.explained_variance_ratio_
                metrics_text = " LDA explained class separability ratio: \n"
                metrics_text += f"{explained_variance}"
                self.metrics_text.setText(metrics_text)
            except:
                self.show_error("You are probably using wrong dataset(regression type dataset)")

        if name == "t-SNE Parameters":
            try:
                n_comp = params.get("n_components", 2)

                # Hata kontrolü: LDA bileşen sayısı en fazla n_classes - 1 olabilir
                if n_comp > 3:
                    self.show_error(
                        f"n_components value could be at most 3  "
                    )
                    return  # Fonksiyondan çık, LDA çalışmasın
                # Parametreleri oku
                #n_comp = params.get("n_components", 2)  # “n_components” olsun
                perplexity = params.get("perplexity", 30)
                random_state = params.get("random_state", 200)

                # t-SNE kısmı
                tsne = TSNE(
                    n_components=n_comp,
                    perplexity=perplexity,
                    random_state=random_state
                )

                # Sadece eğitim verisine fit + transform (görselleştirme için)
                X_train_tsne = tsne.fit_transform(self.X_train)

                # Eğer test verisini de göstermek istersen:
                # X_test_tsne = tsne.fit_transform(self.X_test)
                # Görselleştirme: gerçek etiketleri kullan
                self.update_visualization(self.y_train, X_train_tsne, params=params)
            except:
                self.show_error()

        if name == "K-Means Parameters":
            n_cluster = params.get("n_clusters",1)
            max_iter = params.get("max_iter",1)
            n_init = params.get("n_init",1)
            checkbox = params.get("checkbox",True)

            if checkbox is True:
                pca = PCA(n_components=2)
                X_reduced = pca.fit_transform(self.X_train)
                kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=n_init, max_iter=max_iter)
                y_kmeans = kmeans.fit_predict(X_reduced)
                self.update_visualization(y_kmeans,X_reduced)

            else:
                kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=n_init, max_iter=max_iter)
                y_kmeans = kmeans.fit_predict(self.X_train)
                self.update_visualization(y_kmeans, self.X_train)

    def linear_regression(self, normalize, fit_intercept):
        try:
            selected_method = self.method_combo.currentText()
            # Model oluştur
            lin_reg = LinearRegression(fit_intercept=fit_intercept)

            # Eğer normalize seçildiyse, scaler oluştur
            if normalize:
                scaler = StandardScaler()

                if selected_method == "K-Fold Cross Validation":
                    X_scaled = scaler.fit_transform(self.data.data)
                    y = self.data.target
                else:
                    self.X_train = scaler.fit_transform(self.X_train)
                    self.X_cv = scaler.transform(self.X_cv)
                    self.X_test = scaler.transform(self.X_test)
            else:
                if selected_method == "K-Fold Cross Validation":
                    X_scaled = self.data.data

                    y = self.data.target

            # --- K-FOLD ---
            if selected_method == "K-Fold Cross Validation":

                validation_scores = []
                y_true_all = []
                y_pred_all = []

                for train_index, val_index in self.kfold.split(X_scaled):
                    X_train_fold = X_scaled[train_index]
                    y_train_fold = y[train_index]

                    X_val_fold = X_scaled[val_index]
                    y_val_fold = y[val_index]

                    lin_reg.fit(X_train_fold, y_train_fold)
                    y_val_pred = lin_reg.predict(X_val_fold)

                    y_true_all.extend(y_val_fold)
                    y_pred_all.extend(y_val_pred)

                    mse = mean_squared_error(y_val_fold, y_val_pred)
                    validation_scores.append(mse)
                avg_validation_mse = sum(validation_scores) / len(validation_scores)
                avg_validation_rmse = np.sqrt(avg_validation_mse)

                metrics_text = "K-Fold Regression Metrics:\n\n"
                metrics_text += f"Mean Squared Error (MSE): {avg_validation_mse:.4f}\n"
                metrics_text += f"Root Mean Squared Error (RMSE): {avg_validation_rmse:.4f}"

                self.update_visualization(y_pred_all, y_true_all, params=None)
                self.metrics_text.setText(metrics_text)
            # --- Train CV Test ---

            else:
                # Modeli eğit
                lin_reg.fit(self.X_train, self.y_train)

                # Validation verisiyle tahmin
                y_cv_pred = lin_reg.predict(self.X_cv)
                cv_mse = mean_squared_error(self.y_cv, y_cv_pred)
                cv_rmse = cv_mse ** 0.5

                # Test verisiyle tahmin
                y_test_pred = lin_reg.predict(self.X_test)
                test_mse = mean_squared_error(self.y_test, y_test_pred)
                test_rmse = test_mse ** 0.5
                self.update_visualization(self.y_test, y_test_pred)

                self.update_metrics(y_test_pred)
        except :
            self.show_error("You have to 'Save Selections' first." )

    def logistic_regression(self, C, max_iter, multi_class, normalize):

        try:
            selected_method = self.method_combo.currentText()

            # Model oluştur
            log_reg = LogisticRegression(C=C, max_iter=max_iter, multi_class=multi_class, solver="lbfgs")

            # Normalize işlemi
            if normalize:
                scaler = StandardScaler()

                if selected_method == "K-Fold Cross Validation":
                    X_scaled = scaler.fit_transform(self.data.data)
                    y = self.data.target
                else:
                    self.X_train = scaler.fit_transform(self.X_train)
                    self.X_cv = scaler.transform(self.X_cv)
                    self.X_test = scaler.transform(self.X_test)
            else:
                if selected_method == "K-Fold Cross Validation":
                    X_scaled = self.data.data
                    y = self.data.target

            # --- K-FOLD ---
            if selected_method == "K-Fold Cross Validation":
                validation_scores = []
                y_pred_all = []
                X_test_all = []

                for train_index, val_index in self.kfold.split(X_scaled):
                    X_train_fold = X_scaled[train_index]
                    y_train_fold = y[train_index]
                    X_val_fold = X_scaled[val_index]
                    y_val_fold = y[val_index]

                    log_reg.fit(X_train_fold, y_train_fold)
                    y_val_pred = log_reg.predict(X_val_fold)

                    X_test_all.extend(X_val_fold)
                    y_pred_all.extend(y_val_pred)

                    acc = accuracy_score(y_val_fold, y_val_pred)
                    validation_scores.append(acc)

                avg_validation_acc = sum(validation_scores) / len(validation_scores)

                # Görselleştirme
                self.update_visualization(np.array(y_pred_all), np.array(X_test_all), params={"n_components": 2})
                avg_validation_acc = sum(validation_scores) / len(validation_scores)

                # Görselleştirme
                self.update_visualization(np.array(y_pred_all), np.array(X_test_all), params={"n_components": 2})

                # Metrikler
                metrics_text = "K-Fold Classification Metrics:\n\n"
                metrics_text += f"Average Accuracy: {avg_validation_acc:.4f}\n"

                # Confusion Matrix oluşturmak için gerçek label'lara ihtiyacımız var
                # y_true: tüm fold'lardan birleştirilmiş gerçek etiketler
                y_true_all = []
                for train_index, val_index in self.kfold.split(X_scaled):
                    y_true_all.extend(y[val_index])

                conf_matrix = confusion_matrix(y_true_all, y_pred_all)
                metrics_text += "Confusion Matrix:\n"
                metrics_text += str(conf_matrix)

                self.metrics_text.setText(metrics_text)

            # --- HOLD-OUT ---
            else:
                log_reg.fit(self.X_train, self.y_train)

                y_cv_pred = log_reg.predict(self.X_cv)
                cv_acc = accuracy_score(self.y_cv, y_cv_pred)

                y_test_pred = log_reg.predict(self.X_test)
                test_acc = accuracy_score(self.y_test, y_test_pred)


                self.update_visualization(y_test_pred, self.X_test, params={"n_components": 2})
                self.update_metrics(y_test_pred)

        except Exception as e:
            self.show_error("You have to 'Save Selections' first.")

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

        create_mlp = self.create_mlp_controls()
        mlp_layout.addWidget(create_mlp)

        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 1)

        '''# Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)'''

        '''# Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)

        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_networks)
        mlp_layout.addWidget(train_btn)'''

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
    '''def ya_sabir_amk(self):
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        try:

            batch_size = self.batch_size_spin.value()

            # Create and compile model

            model = self.create_neural_network()
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:

                X_train = self.X_train
                X_test = self.X_test

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
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

            self.show_error(f"Error training neural network: {str(e)}")'''
    def clear_selected_model(self):
        self.model = None


    @pyqtSlot()
    def load_model_from_file(self):
        try:

            fname = QFileDialog.getOpenFileName(
                self,
                "Open File",
                r"MKT3434_2025\saved_models",
                "All Files (*);; Python Files (*.py);; PNG Files (*.png)",
            )
            if fname[0]:  # boş string değilse (yani bir şey seçildiyse). Ve fname bir tuple döndürür ilk şeysi path,
                self.model = tf.keras.models.load_model(fname[0])#ikincisi ise hangi dosya türünde bulduğu


        except Exception as e:
            self.show_error(e)

    def save_model_from_file(self):
        """Yakında h5 formatı deprecated olacakmış, şuan için umrumda mı peki ? Olmamalı bence de."""

        saved_models = "saved_models"
        model_name, ok = QInputDialog.getText(self, "Model Kaydet", "Model adı girin:")


        #self.model.save('{model_name}.h5')
        save_path = os.path.join(saved_models, f"{model_name}.h5")
        try:
            self.model.save(save_path)

        except Exception as e:
            self.show_error(e)

    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout","LSTM","GRU"])
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
            # Tüm önceki parametreleri temizle
            while params_layout.count():
                child = params_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self.layer_param_inputs.clear()

            layer_type = type_combo.currentText()
            #
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

                # L2 Regularization
                l2_checkbox = QCheckBox("Use L2 Regularization")
                #self.layer_param_inputs["use_l2"] = l2_checkbox

                l2_label = QLabel("L2 Lambda:")
                l2_input = QDoubleSpinBox()
                l2_input.setDecimals(6)
                l2_input.setRange(0.000001, 1.0)
                l2_input.setSingleStep(0.0001)
                l2_input.setValue(0.001)
                l2_label.setEnabled(False)
                l2_input.setEnabled(False)

                self.layer_param_inputs["l2_lambda"] = l2_input

                def toggle_l2(value):
                    enabled = l2_checkbox.isChecked()
                    l2_label.setEnabled(enabled)
                    l2_input.setEnabled(enabled)

                l2_checkbox.stateChanged.connect(toggle_l2)

                # Add widgets
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
                params_layout.addWidget(l2_checkbox)
                params_layout.addWidget(l2_label)
                params_layout.addWidget(l2_input)


            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input

                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
            elif layer_type == "LSTM":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1024)
                units_input.setValue(50)
                self.layer_param_inputs["units"] = units_input

                return_seq_checkbox = QCheckBox("Return Sequences")
                return_seq_checkbox.setChecked(True)
                self.layer_param_inputs["return_sequences"] = return_seq_checkbox


                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(return_seq_checkbox)
            elif layer_type == "GRU":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1024)
                units_input.setValue(50)
                self.layer_param_inputs["units"] = units_input

                return_seq_checkbox = QCheckBox("Return Sequences")
                return_seq_checkbox.setChecked(True)
                self.layer_param_inputs["return_sequences"] = return_seq_checkbox

                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(return_seq_checkbox)


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
                elif isinstance(widget, QCheckBox):
                    layer_params[param_name] = widget.isChecked()

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
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.01)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        group.setLayout(layout)
        return group


    def create_cnn_controls(self):
        """Create controls for CNN architecture and training parameters with tabs"""
        group = QGroupBox("CNN Configuration")
        layout = QVBoxLayout()


        # Sekmeli yapı
        tabs = QTabWidget()

        # === 1. Sekme: CNN Architecture ===
        arch_tab = QWidget()
        arch_layout = QVBoxLayout()

        self.cnn_layers_list = QListWidget()
        arch_layout.addWidget(self.cnn_layers_list)
        

        # Optimizer seçimi
        optimizer_layout = QHBoxLayout()
        optimizer_label = QLabel("Optimizer:")
        optimizer_combo = QComboBox()
        optimizer_combo.addItems(["Adam", "SGD", "RMSProp"])
        optimizer_layout.addWidget(optimizer_label)
        optimizer_layout.addWidget(optimizer_combo)
        arch_layout.addLayout(optimizer_layout)


        def on_optimizer_changed(text):
            self.selected_optimizer = text

        optimizer_combo.currentTextChanged.connect(on_optimizer_changed)

        # Butonlar
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        remove_btn = QPushButton("Remove Selected Layer")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        arch_layout.addLayout(btn_layout)

        arch_tab.setLayout(arch_layout)
        tabs.addTab(arch_tab, "CNN Architecture")

        # === 2. Sekme: Training Parameters ===
        train_tab = QWidget()
        train_layout = QVBoxLayout()
        train_layout.addWidget(self.create_training_params_group())  # Burada QGroupBox widget olarak ekleniyor
        train_tab.setLayout(train_layout)
        tabs.addTab(train_tab, "Training Parameters")

        # === Ana layouta sekmeleri ekle ===
        layout.addWidget(tabs)

        # Eğitim başlatma butonu
        train_btn = QPushButton("Train CNN")
        layout.addWidget(train_btn)
        # Load Model ve Clear Model butonları için yatay bir layout oluşturuyoruz
        model_buttons_layout = QHBoxLayout()

        # Load Model butonu
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model_from_file)
        model_buttons_layout.addWidget(load_model_btn)  # Yatay layout'a ekle

        # Clear Existing Model butonu
        clear_model_btn = QPushButton("Clear Existing Model")
        clear_model_btn.clicked.connect(self.clear_selected_model)
        model_buttons_layout.addWidget(clear_model_btn)  # Yatay layout'a ekle (load_model_btn'nin yanına)

        # Oluşturduğumuz yatay layout'u ana dikey layout'a ekliyoruz
        layout.addLayout(model_buttons_layout)
        save_model_btn = QPushButton("Save Model")
        layout.addWidget(save_model_btn)
        save_model_btn.clicked.connect(self.save_model_from_file)
        # Katmanları tut
        self.cnn_layer_config = []

        def on_add_layer():
            self.layer_config = []
            self.add_layer_dialog()
            if self.layer_config:
                layer = self.layer_config[-1]
                self.cnn_layer_config.append(layer)
                self.cnn_layers_list.addItem(f"{layer['type']} - {layer['params']}")

        def on_remove_layer():
            row = self.cnn_layers_list.currentRow()
            if row >= 0:
                self.cnn_layers_list.takeItem(row)
                self.cnn_layer_config.pop(row)

        train_btn.clicked.connect(self.train_neural_networks)
        add_btn.clicked.connect(on_add_layer)
        remove_btn.clicked.connect(on_remove_layer)

        group.setLayout(layout)
        return group

    def create_mlp_controls(self):
        """Create controls for MLP architecture and training parameters with tabs"""
        group = QGroupBox("MLP Configuration")
        layout = QVBoxLayout()

        # Sekmeli yapı
        tabs = QTabWidget()

        # === 1. Sekme: MLP Architecture ===
        arch_tab = QWidget()
        arch_layout = QVBoxLayout()

        self.mlp_layers_list = QListWidget()
        arch_layout.addWidget(self.mlp_layers_list)

        # Optimizer seçimi
        optimizer_layout = QHBoxLayout()
        optimizer_label = QLabel("Optimizer:")
        optimizer_combo = QComboBox()
        optimizer_combo.addItems(["Adam", "SGD", "RMSProp"])
        optimizer_layout.addWidget(optimizer_label)
        optimizer_layout.addWidget(optimizer_combo)
        arch_layout.addLayout(optimizer_layout)

        def on_optimizer_changed(text):
            self.selected_optimizer = text

        optimizer_combo.currentTextChanged.connect(on_optimizer_changed)

        # Butonlar
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Dense Layer")
        remove_btn = QPushButton("Remove Selected Layer")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        arch_layout.addLayout(btn_layout)

        arch_tab.setLayout(arch_layout)
        tabs.addTab(arch_tab, "MLP Architecture")

        # === 2. Sekme: Training Parameters ===
        train_tab = QWidget()
        train_layout = QVBoxLayout()
        train_layout.addWidget(self.create_training_params_group())
        train_tab.setLayout(train_layout)
        tabs.addTab(train_tab, "Training Parameters")

        layout.addWidget(tabs)

        # Eğitim başlatma butonu
        train_btn = QPushButton("Train MLP")
        layout.addWidget(train_btn)

        # Load/Clear/Save Model
        model_buttons_layout = QHBoxLayout()

        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model_from_file)
        model_buttons_layout.addWidget(load_model_btn)

        clear_model_btn = QPushButton("Clear Existing Model")
        clear_model_btn.clicked.connect(self.clear_selected_model)
        model_buttons_layout.addWidget(clear_model_btn)

        layout.addLayout(model_buttons_layout)

        save_model_btn = QPushButton("Save Model")
        save_model_btn.clicked.connect(self.save_model_from_file)
        layout.addWidget(save_model_btn)

        self.mlp_layer_config = []

        def on_add_layer():
            self.layer_config = []
            self.add_layer_dialog(layer_type="Dense")  # 'Dense' türünü belirt
            if self.layer_config:
                layer = self.layer_config[-1]
                self.mlp_layer_config.append(layer)
                self.mlp_layers_list.addItem(f"{layer['type']} - {layer['params']}")

        def on_remove_layer():
            row = self.mlp_layers_list.currentRow()
            if row >= 0:
                self.mlp_layers_list.takeItem(row)
                self.mlp_layer_config.pop(row)


        train_btn.clicked.connect(self.train_neural_networks)
        add_btn.clicked.connect(on_add_layer)
        remove_btn.clicked.connect(on_remove_layer)

        group.setLayout(layout)
        return group

    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Configuration")
        layout = QVBoxLayout()

        # Sekmeli yapı
        tabs = QTabWidget()

        # === 1. Sekme: RNN Architecture ===
        arch_tab = QWidget()
        arch_layout = QVBoxLayout()

        self.rnn_layers_list = QListWidget()
        arch_layout.addWidget(self.rnn_layers_list)

        # RNN seçimi (LSTM, GRU)
        """rnn_type_layout = QHBoxLayout()
        rnn_type_label = QLabel("RNN Type:")
        self.rnn_type_combo = QComboBox()
        self.rnn_type_combo.addItems(["LSTM", "GRU"])
        rnn_type_layout.addWidget(rnn_type_label)
        rnn_type_layout.addWidget(self.rnn_type_combo)
        arch_layout.addLayout(rnn_type_layout)"""

        # Optimizer seçimi
        optimizer_layout = QHBoxLayout()
        optimizer_label = QLabel("Optimizer:")
        optimizer_combo = QComboBox()
        optimizer_combo.addItems(["Adam", "SGD", "RMSProp"])
        optimizer_layout.addWidget(optimizer_label)
        optimizer_layout.addWidget(optimizer_combo)
        arch_layout.addLayout(optimizer_layout)

        def on_optimizer_changed(text):
            self.selected_optimizer = text

        optimizer_combo.currentTextChanged.connect(on_optimizer_changed)

        # Butonlar
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        remove_btn = QPushButton("Remove Selected Layer")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        arch_layout.addLayout(btn_layout)

        arch_tab.setLayout(arch_layout)
        tabs.addTab(arch_tab, "RNN Architecture")

        # === 2. Sekme: Training Parameters ===
        train_tab = QWidget()
        train_layout = QVBoxLayout()
        train_layout.addWidget(self.create_training_params_group())
        train_tab.setLayout(train_layout)
        tabs.addTab(train_tab, "Training Parameters")

        layout.addWidget(tabs)

        train_btn = QPushButton("Train RNN")
        layout.addWidget(train_btn)

        # Load/Save/Clear buttons
        model_buttons_layout = QHBoxLayout()
        load_model_btn = QPushButton("Load Model")
        load_model_btn.clicked.connect(self.load_model_from_file)
        model_buttons_layout.addWidget(load_model_btn)

        clear_model_btn = QPushButton("Clear Existing Model")
        clear_model_btn.clicked.connect(self.clear_selected_model)
        model_buttons_layout.addWidget(clear_model_btn)

        save_model_btn = QPushButton("Save Model")
        save_model_btn.clicked.connect(self.save_model_from_file)
        layout.addLayout(model_buttons_layout)
        layout.addWidget(save_model_btn)



        def on_add_layer():
            self.layer_config = []
            self.add_layer_dialog()
            if self.layer_config:
                layer = self.layer_config[-1]
                self.rnn_layer_config.append(layer)
                self.rnn_layers_list.addItem(f"{layer['type']} - {layer['params']}")

        def on_remove_layer():
            row = self.rnn_layers_list.currentRow()
            if row >= 0:
                self.rnn_layers_list.takeItem(row)
                self.rnn_layer_config.pop(row)

        train_btn.clicked.connect(self.train_neural_network_rnn)
        add_btn.clicked.connect(on_add_layer)
        remove_btn.clicked.connect(on_remove_layer)

        group.setLayout(layout)
        return group

    def train_neural_networks(self):
        """Train the neural network with current configuration"""
        #print(self.cnn_layer_config)
        global optimizer
        if not self.cnn_layer_config and self.model is None:


            self.show_error("Please add at least one CNN layer to the network or be sure you loaded your model")
            return

        try:

            """batch_size = self.batch_size_spin.value()
            print("Batch Size:", batch_size)"""
            # Create and compile model
            if self.model is None:
                self.model = self.create_neural_network()

            # sanırım bu model kaydedilecek baboli

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
            print(y_train[0:5]) #one-hot encoding başarılı

            print(len(y_train))         # 56000

            # Compile model
            if self.selected_optimizer == "Adam":
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            elif self.selected_optimizer == "SGD":
                optimizer = optimizers.SGD(learning_rate=learning_rate)
            elif self.selected_optimizer == "RMSProp":
                optimizer = optimizers.RMSprop(learning_rate=learning_rate)
            else:
                optimizer = optimizers.Adam(learning_rate=learning_rate)
            #optimizer = optimizers[self.selected_optimizer](learning_rate=self.lr_spin.value()) hatalı
            #optimizer = optimizers.Adam(learning_rate=learning_rate) hatalı ama son çare
            self.model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])


            """for layer in self.model.layers:

"""
            #0x0000023E61B26F10 >

            # Train model
            history = self.model.fit(X_train, y_train,
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



        for i, layer_config in enumerate(self.layer_config):
            layer_type = layer_config["type"]
            params = layer_config["params"]

            if layer_type == "Dense":
                model.add(layers.Dense(**params))

            elif layer_type == "Conv2D":
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))

            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())

            elif layer_type == "Flatten":
                model.add(layers.Flatten())

            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))

            elif layer_type == "LSTM":
                model.add(layers.LSTM(input_shape=self.X_train.shape[1:]),**params)

            elif layer_type == "GRU":
                if len(model.layers) == 0:
                    # İlk katman, input_shape ekle
                    params["input_shape"] = self.X_train.shape[1:]
                model.add(layers.GRU(**params))
        # Add output layer
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))



        return model

    def train_neural_network_rnn(self):
        """Train the RNN model with current configuration"""
        global optimizer

        if not self.rnn_layer_config and self.model is None:
            self.show_error("Please add at least one RNN layer to the network")
            return
        try:
            # Create and compile model

            model = self.create_neural_network()

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()

            # Optimizer seçimi
            if self.selected_optimizer == "Adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif self.selected_optimizer == "SGD":
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif self.selected_optimizer == "RMSProp":
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # Train model
            """self.y_train = to_categorical(self.y_train, num_classes=2)
            self.y_test = to_categorical(self.y_test, num_classes=2)"""
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

    def update_visualization(self, y_pred, X_test, params=None):
        """Update the visualization with current results"""
        self.figure.clear()

        # Create appropriate visualization based on data
        if len(np.unique(y_pred)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(y_pred, X_test, alpha=0.6)
            ax.plot([min(y_pred), max(y_pred)],
                    [min(y_pred), max(y_pred)],
                    'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")

        else:  # Classification
            if params is not None:
                # GUI'den gelen n_components ve whiten kullanılıyor!
                n_components = params.get("n_components", 2)

                if n_components == 1:
                    # n_components = 1 için histogram kullan
                    ax = self.figure.add_subplot(111)
                    ax.hist(X_test[:, 0], bins=20)
                    ax.set_xlabel("Component 1")
                    ax.set_ylabel("Frequency")

                elif n_components == 2:
                    # n_components = 2 için 2D scatter plot
                    #X_vis = pca.fit_transform(X_test)
                    ax = self.figure.add_subplot(111)
                    scatter = ax.scatter(X_test[:, 0], X_test[:, 1],
                                         c=y_pred, cmap='viridis')
                    self.figure.colorbar(scatter)
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")

                elif n_components == 3:
                    # n_components = 3 için 3D scatter plot
                    #X_vis = pca.fit_transform(X_test)
                    ax = self.figure.add_subplot(111, projection='3d')
                    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2],
                                         c=y_pred, cmap='viridis')
                    self.figure.colorbar(scatter)
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_zlabel("Principal Component 3")

            else:
                # Eğer parametreler gelmediyse, doğrudan 2D scatter plot
                X_vis = X_test
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1],
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
            #r2 = self.current_model.score(self.y_test, y_pred)

            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            #metrics_text += f"R² Score: {r2:.4f}"


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
