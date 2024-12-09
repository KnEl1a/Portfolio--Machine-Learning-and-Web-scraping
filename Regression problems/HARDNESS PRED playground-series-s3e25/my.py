
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, probplot, skew
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import ElasticNetCV
import warnings
from IPython.display import display
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor


# Ignorar advertencias
warnings.filterwarnings("ignore")

def dirp(classObject):
    """
    funcion para ver los metodos publicos de una clase
    """
    # Obtener todos los atributos y métodos
    all_attributes = dir(classObject)

    # Filtrar los atributos que no comienzan con '__'
    public_attributes = [attr for attr in all_attributes if not attr.startswith('__')]

    display(print(public_attributes))

    #dirp(eda)

class my_EDA:
    """
    Esta clase muestra información sobre el dataset
    """
    def __init__(self):
        print()
        print("Object my_EDA created")
        
        print()
    
    def unique_values(self, df):
        b = pd.DataFrame()
        b['N unique value'] = df.nunique()
        b['dtype'] = df.dtypes
        return b
    
    def get_missing_values(self, data, list_mode=False):
        """
        Esta función encuentra los valores faltantes en el dataset
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            Los datos sobre los que deseas ver información
        
        Returns
        ----------
        Una Serie de Pandas que contiene los valores faltantes en orden descendente
        """
        # Obtener la suma de todos los valores faltantes en el dataset
        missing_values = data.isnull().sum()
        # Ordenar los valores faltantes en una Serie de pandas
        missing_values = missing_values.sort_values(ascending=False)
        if list_mode ==True:
            # Retornar la Serie de valores faltantes
            missing_values = list(missing_values[missing_values>0].index)
            
        return missing_values
    
    def info(self, data):
        """
        Esta función muestra información sobre los datos como 
        nombres de características, tipo de datos, número de valores faltantes para cada característica 
        y diez muestras de cada característica
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            Los datos sobre los que deseas ver información
        
        Returns
        ----------
        Información sobre el DataFrame
        """
        self.data = data
        feature_dtypes = self.data.dtypes
        self.missing_values = self.get_missing_values(self.data)
        feature_names = self.missing_values.index
        missing_values = self.missing_values.values
        rows, columns = data.shape

        print("=" * 50)
        print('====> This data contains {} rows and {} columns'.format(rows,columns))
        print("=" * 50)
        print()
        
        print("{:13} {:13} {:30} {:15}".format('Feature Name'.upper(),
                                               'Data Format'.upper(),
                                               'Null values(Num-Perc)'.upper(),
                                               'Seven Samples'.upper()))
        print("{:13} {:13} {:30} {:15}".format('------------',
                                               '-----------',
                                               '---------------------',
                                               '-------------'))
        for feature_name, dtype, missing_value in zip(feature_names,feature_dtypes[feature_names],missing_values):
            print("{:15} {:14} {:20}".format(feature_name,
                                             str(dtype), 
                                             str(missing_value) + ' - ' + 
                                             str(round(100*missing_value/sum(self.missing_values),3))+' %'), end="")

            for i in np.random.randint(0,len(data),7):
                print(data[feature_name].iloc[i], end=",")
            print()

        print("="*50)
        
        print()
        print("Data describe: ")
        print("="*70)
        display(self.data.describe().T)
        print("="*70)
        print()
    def plot_variable_distribution(self, y, message='', use_yeo_johnson=False):
        """
        Función para graficar: histograma y QQ plot para ver la distribución de una variable.

        Parámetros:
            y (pd.Series): Serie de pandas con los datos de la variable.
            message (str): Mensaje opcional sobre la necesidad de conversión.
            use_yeo_johnson (bool): Si True, aplica una transformación Yeo-Johnson a la variable.
        """
        # Verificar si se requiere aplicar la transformación Yeo-Johnson
        if use_yeo_johnson:
            from sklearn.preprocessing import PowerTransformer
            
            # Crear una instancia de PowerTransformer para Yeo-Johnson
            yj_transformer = PowerTransformer(method='yeo-johnson')
            
            # Aplicar la transformación Yeo-Johnson
            data = yj_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
        else:
            # Si no se solicita transformación, utilizar los datos tal cual
            data = y.dropna()

        # Crear figura
        f, (ax_left, ax_right1, ax_right2) = plt.subplots(1, 3, figsize=(20, 5), facecolor='whitesmoke')

        # Información sobre la variable
        ax_left.axis('off')
        ax_left.text(0.5, 0.9, 'Variable', color='crimson', size=22, weight='bold', ha='center')
        ax_left.text(0.5, 0.7, f'Skew: {pd.Series(data).skew():.2f}', size=20, ha='center')
        ax_left.text(0.5, 0.6, f'Kurt: {pd.Series(data).kurt():.2f}', size=20, ha='center')
        ax_left.text(0.5, 0.5, f'Missing count: {y.isnull().sum()}', size=20, ha='center')
        ax_left.text(0.5, 0.3, f'{message}', color='crimson', size=20, ha='center')

        # Histograma con ajuste normal
        sns.histplot(data, kde=True, stat='density', color='crimson', ax=ax_right1)

        # Ajustar distribución normal
        mu, std = norm.fit(data)
        xmin, xmax = ax_right1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax_right1.plot(x, p, 'k', linewidth=2)

        ax_right1.spines[['top', 'right']].set_visible(False)
        ax_right1.set_title('Histogram', color='crimson', weight='bold', size=15)

        # QQ plot
        probplot(data, dist="norm", plot=ax_right2)
        ax_right2.spines[['top', 'right']].set_visible(False)
        ax_right2.set_title('QQ Plot', color='crimson', weight='bold', size=15)

        # Ajustar el diseño y mostrar la figura
        plt.tight_layout()
        plt.show()
        
    # def dependent_var(self, df, y, message='', log_trans=False):
    #     """
    #     Función para graficar: histograma y QQ plot para ver la distribución de la variable objetivo.

    #     Parámetros:
    #         df (pd.DataFrame): DataFrame con los datos.
    #         y (str): Nombre de la columna de la variable objetivo.
    #         message (str): Mensaje opcional sobre la necesidad de conversión.
    #         log_trans (bool): Si True, aplica una transformación logarítmica a la variable objetivo.
    #     """
    #     # Aplicar transformación logarítmica si se solicita
    #     if log_trans:
    #         data = df[y].dropna().apply(lambda x: x if x > 0 else np.nan).apply(np.log1p).dropna()
    #     else:
    #         data = df[y].dropna()

    #     # Crear figura
    #     f, (ax_left, ax_right1, ax_right2) = plt.subplots(1, 3, figsize=(20, 5), facecolor='whitesmoke')

    #     # Información sobre la variable
    #     ax_left.axis('off')
    #     ax_left.text(0.5, 0.9, y, color='crimson', size=22, weight='bold', ha='center')
    #     ax_left.text(0.5, 0.7, f'Skew: {data.skew():.2f}', size=20, ha='center')
    #     ax_left.text(0.5, 0.6, f'Kurt: {data.kurt():.2f}', size=20, ha='center')
    #     ax_left.text(0.5, 0.5, f'Missing count: {df[y].isnull().sum()}', size=20, ha='center')
    #     ax_left.text(0.5, 0.3, f'{message}', color='crimson', size=20, ha='center')

    #     # Histograma con ajuste normal
    #     sns.histplot(data, kde=True, stat='density', color='crimson', ax=ax_right1)

    #     # Ajustar distribución normal
    #     mu, std = norm.fit(data)
    #     xmin, xmax = ax_right1.get_xlim()
    #     x = np.linspace(xmin, xmax, 100)
    #     p = norm.pdf(x, mu, std)
    #     ax_right1.plot(x, p, 'k', linewidth=2)

    #     ax_right1.spines[['top', 'right']].set_visible(False)
    #     ax_right1.set_title('Histogram', color='crimson', weight='bold', size=15)

    #     # QQ plot
    #     probplot(data, dist="norm", plot=ax_right2)
    #     ax_right2.spines[['top', 'right']].set_visible(False)
    #     ax_right2.set_title('QQ Plot', color='crimson', weight='bold', size=15)

    #     # Ajustar el diseño y mostrar la figura
    #     plt.tight_layout()
    #     plt.show()

        # # Llamar a la función
        # data = dependent_var(train, 'SalePrice', message='not', log_trans=True)
        # original_data = np.expm1(data) # re conversion
        # original_data

        # return data
    
    # def vars_list(self, df, verbose=-1):
    #     """
    #     VERBOSE = predefinicdo: -1 (no mostrar), MOSTRAR listas de variables categoricas y numericas.
    #     INPUT : pandas dataframe
    #     OUTPUT: listado de variables numericas y listado de variables categoricas 
    #      devuelve listas de 
    #     """
    #     num_vars = df.columns[df.dtypes != 'object']
    #     obj_vars = df.columns[df.dtypes == 'object']
    #     if verbose != -1
    #         print('\nNumerical vars: ')
    #         print(num_vars.values)
    #         print('\nObject vars: ')
    #         print(obj_vars.values)
    #     return num_vars, obj_vars
    
    def vars_list(self, df, verbose=-1):
            """
            Muestra listas de variables categóricas y numéricas.
            
            :param df: pandas DataFrame
            :param verbose: si es distinto de -1, muestra las listas de variables categóricas y numéricas
            :return: tupla de listas de variables 1)numéricas y 2)categóricas
            """
            num_vars = df.columns[df.dtypes != 'object'].tolist()
            obj_vars = df.columns[df.dtypes == 'object'].tolist()
            
            if verbose != -1:
                print('\nNumerical vars:')
                print(num_vars)
                print('\nObject vars:')
                print(obj_vars)
            return num_vars, obj_vars
    
    
    def exploration_num(self, df):
        j = 1
        a = (len(df.select_dtypes(exclude='object').columns)//3)+1
        plt.figure(figsize=(20,a*5))
        for i in df.select_dtypes(exclude='object'):
            plt.subplot(a,3,j)
            sns.distplot(df[i])
            plt.axvline(df[i].min(),c='b',label='min')
            plt.axvline(df[i].quantile(.25),c='orange',label='25%')
            plt.axvline(df[i].median(),c='y',label='median')
            plt.axvline(df[i].mean(),c='g',label='mean')
            plt.axvline(df[i].quantile(.75),c='brown',label='75%')
            plt.axvline(df[i].max(),c='r',label='max')
            j=j+1

        plt.legend()
        
    def exploration_cat(self, df, y):
        """ df: pd data frame
            y: string. is the name of the target variable. example : 'SalePrice'
        """
        a = (len(df.select_dtypes(include='object').columns)//3)+1
        j=1
        plt.figure(figsize=(20,a*5))
        for i in df.select_dtypes(include='object'):
            plt.subplot(a,3,j)
            sns.violinplot(y=df[y],x=df[i])
            j=j+1
            
    def plot_high_corr_heatmap(self, df, y):
        """
        Esta función crea un mapa de calor de las variables con mayor correlación con la variable objetivo 'y'.

        Parámetros:
        df : pandas DataFrame
            El conjunto de datos que contiene las variables predictoras y la variable objetivo.
        y : str
            El nombre de la columna que contiene la variable objetivo.

        Retorno:
        Muestra un gráfico de mapa de calor de las variables con mayor correlación con la variable objetivo.
        """
        # Verifica que la variable objetivo esté en el DataFrame
        if y not in df.columns:
            raise ValueError(f"La columna '{y}' no se encuentra en el DataFrame.")

        # Selecciona solo las columnas numéricas
        num_vars = df.select_dtypes(include=['float64', 'int64']).columns

        # Asegúrate de que la variable objetivo sea numérica
        if y not in num_vars:
            raise ValueError(f"La columna '{y}' no es numérica.")

        # Filtra el DataFrame para que contenga solo las columnas numéricas
        df_num = df[num_vars]

        # Obtener las 7 variables más correlacionadas con la variable objetivo
        corr = df_num.corr()
        highcorr_vars = abs(corr[y]).sort_values(ascending=False).index[:13]

        # Crear el gráfico
        f, ax = plt.subplots(figsize=(10, 7))

        # Crear el mapa de calor de las variables más correlacionadas
        sns.heatmap(df_num[highcorr_vars].corr(), annot=True, ax=ax, cmap='coolwarm')

        # Mostrar el gráfico
        plt.show()
        return highcorr_vars

        # Ejemplo de uso
        #plot_high_corr_heatmap(train, 'SalePrice')
    

    def plot_missing(self, df):
        df = pd.concat((train, test)).drop(['SalePrice'], axis=1)
        cnt_missing = all_data.isnull().sum().sort_values(ascending=False)
        cnt_percent = cnt_missing / df.shape[0] * 100
        missing_table = pd.DataFrame([cnt_missing, cnt_percent], 
                                     index=['missing count', 'missing percent']).T
        missing_table = missing_table[missing_table['missing count'] > 0]
        missing_table = missing_table.reset_index()
        missing_table['missing count'] = missing_table['missing count'].astype(int)
        color_list=[['whitesmoke', 'white', 'white']]

        fig = plt.figure(facecolor='whitesmoke')
        ax1 = fig.add_axes([0, 0, 1, 0.1]) 
        ax2 = fig.add_axes([1.5, -2.3, 1, 2.3], facecolor='whitesmoke') 

        ax2.spines[['top', 'right']].set_visible(False)

        ax1.set_axis_off()

        table=ax1.table(cellText = missing_table.values[:20], colLabels=missing_table.columns,
                          colColours=['crimson']*3, cellColours=color_list*20)
        table.auto_set_font_size(False) 
        table.set_fontsize(16)  
        table.scale(1.5, 2.7) 
        ax1.text(0.67, .9, 'Missing count and percent', color='crimson', fontsize=20, fontweight='bold')
        ax1.text(1.4, .9, 'by values', fontsize=20, fontweight='bold')

        sns.barplot(y=missing_table['index'], x=missing_table['missing percent'], orient = "h", ax=ax2)
        plt.show()

        #ejemplo de uso
        #plot_missing(all_data)
        
    def hypo_test(self, df ,x, y, cat=False):
        """
        INPUT.
        df: pandas dataframe
        x: str type. exp: 'OveralQual'
        y: str type. exp: 'SalePrice'
        
        OUTPUT
        plt. QQplot scatter
        
        MODE: IF cat = True. shows a catplot. en lugar de un scatter plot
        """
        df = train
        f, ax = plt.subplots(1, 4, figsize=(25, 5), facecolor='whitesmoke')
        if cat:
            sns.boxplot(x=train[x], y=train[y], ax=ax[0], color='crimson')
        else:
            sns.scatterplot(x=train[x], y=train[y], ax=ax[0], color='crimson')
            sns.regplot(x=train[x], y=train[y], ax=ax[0], color='crimson')
        sns.residplot(x=train[x], y=train[y], ax=ax[1], color='crimson')
        sns.distplot(train[x], fit=norm, ax=ax[2], color='crimson')
        probplot(train[x], plot=ax[3])
        ax[0].set_facecolor('whitesmoke')
        ax[1].set_facecolor('whitesmoke')
        ax[2].set_facecolor('whitesmoke')
        ax[3].set_facecolor('whitesmoke')
        ax[0].spines[['top', 'right']].set_visible(False)
        ax[1].spines[['top', 'right']].set_visible(False)
        ax[2].spines[['top', 'right']].set_visible(False)
        ax[3].spines[['top', 'right']].set_visible(False)

        f.suptitle(f'{x}', color='crimson', weight='bold', size=20)

        plt.show()
        
    def create_features(self,df, columns):
        """
        Crea nuevas características en un DataFrame basadas en transformaciones polinómicas y raíces cuadradas
        de las columnas especificadas.

        :param df: DataFrame en el que se agregarán nuevas características.
        :param columns: Lista de nombres de columnas para generar nuevas características.
        :return: DataFrame con nuevas características agregadas.
        """
        new_df = df.copy()  # Crear una copia del DataFrame original para evitar modificarlo

        for col in columns:
            if col in new_df.columns:
                new_df[f"{col}-2"] = new_df[col] ** 2
                new_df[f"{col}-3"] = new_df[col] ** 3
                new_df[f"{col}-Sq"] = np.sqrt(new_df[col])
            else:
                print(f"Column '{col}' not found in DataFrame.")

        return new_df

    def scarplot_strongs_vars(self, df, y, columns):
        """
        df: pandas dataframe exp. train
        y: str. exp: 'SalePrice'
        columns: list. str. exp: cors = ['wert', 'dsfg', 'xcvb', 'adsf', 'cbvn', 'bnv']
        """
        # Lista de nombres de columnas
        train = df
        column_names = list(columns)

        # Crear una cuadrícula de gráficos con 2 columnas y el número adecuado de filas
        num_plots = len(column_names)
        nrows = (num_plots + 1) // 2  # Calcular el número de filas necesarias
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, nrows * 4))

        # Iterar sobre las columnas y graficar en cada subplot
        for idx, col in enumerate(column_names):
            row = idx // 2  # Fila actual
            col_pos = idx % 2  # Columna actual (0 o 1)

            ax = axes[row, col_pos]  # Seleccionar el eje adecuado
            ax.scatter(train[col], train[y], c="crimson", marker="s", s=10)
            ax.set_title(f"{col} vs {y}")
            ax.set_xlabel(col)
            ax.set_ylabel(y)
            # Configurar la cuadrícula con transparencia
            ax.grid(True, alpha=0.6)  # Ajusta el valor de alpha para la transparencia


        # Ajustar el diseño para que no se superpongan los gráficos
        plt.tight_layout()
        plt.show()
    
    
    
        

# train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col = "Id")
# test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col = "Id")
# all_data = pd.concat([train,test], axis = 0)

# all_data = all_data.drop('SalePrice', axis=1)

# # Crear una instancia de la clase
# eda = my_EDA()

# # Llamar al método info en la instancia
# eda.info(all_data)


# eda.dependent_var(train, 'SalePrice', log_trans=False, message='need normalization')
# y = eda.dependent_var(train, 'SalePrice', message='not needs normalization', log_trans=True)




class my_preproc:
    """
    This class prepares the data berfore applying ML
    """
    def __init__(self):
        
        print()
        print('pre-processing object is created')
        print()        
    
    def add_mathematical_transformed_cols(df, columns):
        """
        Añade nuevas características al DataFrame para cada columna en 'columns'.
        Las nuevas características son: la columna al cuadrado, al cubo, y su raíz cuadrada.

        Parámetros:
        - df: pd.DataFrame, DataFrame de entrada.
        - columns: list de str, lista de nombres de columnas a transformar.

        Retorna:
        - pd.DataFrame con las nuevas características añadidas.
        """
        df_transformed = df.copy()  # Crear una copia del DataFrame original

        for col in columns:
            if col in df.columns:
                # Crear nuevas características
                df_transformed[col + '_squared'] = df[col] ** 2
                df_transformed[col + '_cubed'] = df[col] ** 3
                df_transformed[col + '_sqrt'] = np.sqrt(df[col].clip(lower=0))  # Aplicar sqrt solo a valores no negativos

        return df_transformed
    
    def fill(self, data, columns, fill_with):
        """
        INPUT->
        data : pandas df. exp-> all_data
        columns: list of columns. exp: ["sdf","rty","tyu","dsfsd","ñlk"]
        fill_with: str type. exp: 'Zero'
        
        MODES of fill_with: zero/0 , mode/(moda), mean, median
        
        OUTPUT->
        data: a pandas dataframe
        
        """
        
        column = columns 
        if str(fill_with).lower() in ['zero', 0]:
            data[column] = data[column].fillna(0)
        elif str(fill_with).lower()in ['mode', 'moda']:
            data[column] = data[column].fillna(self.data[column].mode()[0])
        elif str(fill_with).lower()=='mean':
            data[column] = data[column].fillna(self.data[column].mean())
        elif str(fill_with).lower()=='median':
            data[column] = data[column].fillna(self.data[column].median())
        else:
            data[column] = data[column].fillna(fill_with)
        return data    
   
    def label_encoder(self, data, columns):
        data = data
        """
        This function is used to encode the data to categorical values to benefit from increasing or 
        decreasing to build the model    
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to encode.
        columns : columns to convert.
        
        Returns
        ----------
        A dataset without categorical data.
        """

        # Convert all categorical collumns to numeric values
        lbl = LabelEncoder() 
        
        data[columns] = data[columns].apply(lambda x:lbl.fit_transform(x.astype(str)).astype(int))
        
        return data 
    
    def get_dummies(self,data , columns):
        data = data
        """
        This function is used to convert the data to dummies values.
        ...
        Attributes
        ----------
        data : Pandas DataFrame
            The data you want to convert.
        
        Returns
        ----------
        A dataset with dummies.
        """
        
        # convert our categorical columns to dummies
        for col in columns:
            dumm = pd.get_dummies(data[col], prefix = col, dtype=int)
            data = pd.concat([data, dumm], axis=1)

        data.drop(columns, axis=1, inplace=True)
        
        return data
        
    def scaler_data_Standard(self, data, columns):
        data = data
        """
        Escala los datos utilizando la estandarización (Z-score normalization).
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(self.data[columns])

        return data
   
    def scaler_data_Robust(self, data, columns): # tenemos la opcion pero, podemos hacerlo en la pipeline
        data = data
        """
        Escalamos los datos utilizando RobustScaler. Esto lo usamos en caso de valores atipicos
        """
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        data[columns] = scaler.fit_transform(data[columns])

        return data
    

    def list_skew_num(self, df, SKEW = 0.75, positive = True):
        """ df: de pandas
            SKEW: valor minimo
        """
        numeric_feats = df.dtypes[all_data.dtypes != "object"].index

        # Check the skew of all numerical features
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        print("\nSesgo. Skewness: \n")
        display(skewness.head(12))
        
        if positive == True:
            skewness = skewness[abs(skewness) > SKEW]
        else: 
            skewness = skewness[abs(skewness) < SKEW]
        print("There are {} skewed numerical features".format(skewness.shape[0]))

        skewness = skewness.dropna()
        skewness = list(skewness.index)
        # mn = list(skewness.index)
        # mn

        #ejemplo de uso:
        #list_skew_num(all_data)
        return skewness

        # skewness = list_skew_num(all_data, 0.79)
        # skewness
        
    def apply_embeddings(self, df, categorical_vars, embedding_dim):
        """
        exp.
        ---------
        # Ejemplo de DataFrame con varias columnas categóricas
        data = {'category1': ['cat are you', 'dog is your mom ', 'fishz zxcv zcv x', 'dog asdf as dsf sd ', 'cat eq rqew r', 'fish u rtyu ', 'cat pogfphgf  '],
                'category2': ['red cxvb xcv  cxb ', 'blue rewt rew wert  ', 'green adsfds ', 'blueasdfadsf ', 'greends ', 'redds  ', 'bluesad'],
                'category3': ['cat are you', 'cat are you ', 'cat are you', 'cat are you ', 'cat eq rqew r', 'fish u rtyu ', 'cat pogfphgf  '],
                'category4': ['cat are you ', 'cat are you  ', 'green adsfds ', 'blueasdfadsf ', 'greends ', 'redds  ', 'bluesad']}
        df = pd.DataFrame(data)

        # Lista de columnas categóricas
        categorical_vars = ['category1', 'category2', "category3", "category4"]

        # Aplicar embeddings con dimensión de 8
        embedding_dim = 8
        df_with_embeddings = apply_embeddings(df, categorical_vars, embedding_dim)

        # print("DataFrame con embeddings:\n", df_with_embeddings)
        display(df_with_embeddings)
        display(df_with_embeddings.loc[:,"category3_emb_0":])
        
        
        how it works
        -------------
        Apply embeddings to categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing categorical columns.
            categorical_vars (list of str): List of categorical column names.
            embedding_dim (int): Dimension of the embedding vectors.

        Returns:
            pd.DataFrame: DataFrame with embeddings applied to the categorical columns.
        """
        # Crear un diccionario para almacenar los embeddings por columna
        embeddings_dict = {}

        for col in categorical_vars:
            # Paso 1: Convertir categorías a índices numéricos
            label_encoder = LabelEncoder()
            df[col + '_encoded'] = label_encoder.fit_transform(df[col])

            # Parámetros del embedding
            input_dim = len(label_encoder.classes_)  # Número de categorías únicas

            # Paso 2: Definir el modelo de embeddings
            inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)
            embedding_layer = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=embedding_layer)

            # Inicializar los embeddings aleatoriamente
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Paso 3: Obtener embeddings
            input_data = np.array(df[col + '_encoded'])
            embeddings = model.predict(input_data)

            # Reemplazar columna original con embeddings
            embedding_df = pd.DataFrame(embeddings.reshape(-1, embedding_dim), columns=[f'{col}_emb_{i}' for i in range(embedding_dim)])
            df = pd.concat([df, embedding_df], axis=1)
            df.drop(columns=[col, col + '_encoded'], inplace=True)

        return df
    

    # Datos de entrenamiento (suponiendo que `train` es tu DataFrame estandarizado)
    # train = np.random.rand(100, 2)  # Ejemplo de datos, reemplaza con tus datos reales
    def best_k_plot(self, X,minn=2, maxx=15):
        """
        EXP : best_k_plot(train)
        X: pandas dataframe
        minn: int.
        maxx. int.
        """
        train = X
        # Definir rangos para K
        K_range = range(minn,maxx)

        # Almacenar resultados
        inertia = []
        silhouette_scores = []
        models = {}  # Diccionario para almacenar los modelos ajustados

        # Entrenar KMeans para cada valor de K
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(train)
            models[k] = kmeans  # Guardar el modelo ajustado
            inertia.append(kmeans.inertia_)
            labels = kmeans.predict(train)  # Predecir las etiquetas
            silhouette_scores.append(silhouette_score(train, labels))

        # Crear subgráficos
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Gráfico del Codo
        ax1.plot(K_range, inertia, 'bo-', marker='o', linestyle='-', color='g')
        ax1.set_ylabel('Suma de Distancias Cuadradas Dentro del Clúster')
        ax1.set_title('Método del Codo')
        ax1.grid(True)

        # Gráfico de Silhouette Score
        ax2.plot(K_range, silhouette_scores, 'bo-', marker='o', linestyle='-', color='b')

        # Añadir líneas verticales
        for k in K_range:
            ax2.axvline(x=k, linestyle='--', color='gray', alpha=0.5)

        # Ajustar los números en el eje X
        ax2.set_xticks(np.arange(2, 15, 1))  # Cambia el intervalo si es necesario
        ax2.set_xlabel('Número de Clusters')
        ax2.set_ylabel('Puntuación de Silhouette')
        ax2.set_title('Método del Silhouette Score')
        ax2.grid(True)

        # Mostrar gráficos
        plt.tight_layout()
        plt.show()


        #best_k_plot(train)
    def column_cluster__Kmeans(self, df, N_CLUSTER = 6, cluster_type = 'category'):
        """
        df: padnas dtaframe
        Ncluster: 6
        cluster_type: category
        """
        # Ajustar el modelo KMeans
        kmeans = KMeans(n_clusters=N_CLUSTER, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df)

        # Convertir la columna 'Cluster' a tipo categórico
        df['Cluster'] = df['Cluster'].astype(cluster_type)
        return df


class my_ML:
    """
    ML  class . Predictive models
    """
    def __init__(self):
        
        print()
        print('models ML is created')
        print()  
    
    
    def r2_cv_train(self, model):
        # Calcula el R^2 usando cross_val_score y la métrica 'r2'
        r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
        return r2

    def r2_cv_test(self, model):
        # Calcula el R^2 usando cross_val_score y la métrica 'r2'
        r2 = cross_val_score(model, X_test, y_test, scoring='r2', cv=5)
        return r2
    
    def obtener_datos_reg(self, X,y, train_size=0.8, random_state=1, train_valid_test_MODE = False):
        
        if train_valid_test_MODE == True:
            # Primera división: conjunto de prueba
            X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

            # Segunda división: conjunto de validación
            X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, train_size=train_size, random_state=random_state)

            total = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]

            def porcent(num):
                result = num/total * 100 
                return f"{int(result)}%"

            print("el total de filas es : ", total)
            print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}  -> {porcent(X_train.shape[0])}")
            print(f"Tamaño del conjunto de validación: {X_valid.shape[0]} -> {porcent(X_valid.shape[0])}")
            print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} -> {porcent(X_test.shape[0])}")
            return X_train, X_valid, y_train, y_valid, X_test, y_test
        else:

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_size, random_state=random_state)

            total = X_train.shape[0] + X_valid.shape[0]

            def porcent(num):
                result = num/total * 100 
                return f"{int(result)}%"

            print("el total de filas es : ", total)
            print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}  -> {porcent(X_train.shape[0])}")
            print(f"Tamaño del conjunto de validación: {X_valid.shape[0]} -> {porcent(X_valid.shape[0])}")
            return X_train, X_valid, y_train, y_valid
        

    def Auto_xgbrCV(self, X_train, y_train, N_trials = 50):
        """ AUTOMATICAMENTE ENTRENAMOS UNA XGBREGRESOSOR
            INPUT: 
            X_train, y_train. pandas dataframe como los que nos data split
            N_trials=50. int type.  
        """
        def objective(trial):
            xgb_params = dict(
                max_depth=trial.suggest_int("max_depth", 2, 14),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
                n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
                subsample=trial.suggest_float("subsample", 0.2, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
                tree_method='gpu_hist',  # Usar GPU para el entrenamiento
            )

            xgb = XGBRegressor(**xgb_params)

            # Validación cruzada para evaluar el modelo
            scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            return -np.mean(scores)

        # Crear el estudio de Optuna usando TPE
        study = optuna.create_study(sampler=TPESampler(), direction="minimize")

        # Optimizar el estudio
        study.optimize(objective, n_trials=N_trials)

        # Obtener los mejores parámetros
        xgb_params = study.best_params
        print(f"Best parameters: {xgb_params}")

        xgb_new=XGBRegressor(**xgb_params)
        return xgb_new

    def Auto_ElasticNetCV(self, X_train, y_train, X_test, y_test,Ynorm_mode = True):
        """ LO DEJAMOS YA PARA LO ULTIMO cuando nuestra baseline es considerada optima
            ENCONRTAMOS LOS HYPERPARAMETROS AUTOMATICAMENTE DE AUOTO ELASTICNET
            INPUT.
            Ynorm_mode: bool. sirve para re-transformar todo de log1p a exp1m con numpy.

        """
        # Define error measure for official scoring : RMSE
        scorer = make_scorer(mean_squared_error, greater_is_better=False)



        # Initial search for ElasticNet parameters
        elasticNet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                                  alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                          0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                                  max_iter=50000, cv=5)
        elasticNet.fit(X_train, y_train)

        alpha = elasticNet.alpha_
        ratio = elasticNet.l1_ratio_
        print("Best l1_ratio :", ratio)
        print("Best alpha :", alpha)

        # Refine search around the best l1_ratio
        elasticNet = ElasticNetCV(l1_ratio=[ratio * 0.85, ratio * 0.9, ratio * 0.95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                                  alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                          0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                                  max_iter=50000, cv=5)
        elasticNet.fit(X_train, y_train)

        if elasticNet.l1_ratio_ > 1:
            elasticNet.l1_ratio_ = 1
        alpha = elasticNet.alpha_
        ratio = elasticNet.l1_ratio_
        print("Best l1_ratio :", ratio)
        print("Best alpha :", alpha)

        # Further refine search around the best alpha
        elasticNet = ElasticNetCV(l1_ratio=[ratio],
                                  alphas=[alpha * 0.6, alpha * 0.65, alpha * 0.7, alpha * 0.75, alpha * 0.8, alpha * 0.85, alpha * 0.9, 
                                          alpha * 0.95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 
                                          alpha * 1.35, alpha * 1.4], 
                                  max_iter=50000, cv=5)
        elasticNet.fit(X_train, y_train)

        alpha = elasticNet.alpha_
        ratio = elasticNet.l1_ratio_
        print("Best l1_ratio :", ratio)
        print("Best alpha :", alpha)

        # Evaluate and plot results
        print("ElasticNet R2 on Training set :", r2_cv_train(elasticNet).mean())
        print("ElasticNet R2 on Test set :", r2_cv_train(elasticNet).mean())


        if Ynorm_mode == False:
            y_train_ela = elasticNet.predict(X_train)
            y_test_ela = elasticNet.predict(X_test)

            # Plot predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(y_train, y_train_ela, c="blue", marker="s", label="Training data")
            plt.scatter(y_test, y_test_ela, c="lightgreen", marker="s", label="Validation data")
            plt.title("Linear regression with ElasticNet regularization")
            plt.xlabel("Real values")
            plt.ylabel("Predicted values")
            plt.legend(loc="upper left")
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c="red", linestyle="--")
            plt.show()

        if Ynorm_mode == True:

            y_train_ela = elasticNet.predict(X_train)
            y_test_ela = elasticNet.predict(X_test)

            y_train_ela = np.expm1(y_train_ela)
            y_test_ela = np.expm1(y_test_ela)

            y_train = np.expm1(y_train)
            y_test = np.expm1(y_test)

            # Plot predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(y_train, y_train_ela, c="blue", marker="s", label="Training data")
            plt.scatter(y_test, y_test_ela, c="lightgreen", marker="s", label="Validation data")
            plt.title("Linear regression with ElasticNet regularization")
            plt.xlabel("Real values")
            plt.ylabel("Predicted values")
            plt.legend(loc="upper left")
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], c="red", linestyle="--")
            plt.show()

        return elasticNet

    def cross_validate_kfold_v1(self, X, y, tf_mode=False, log=True, n_splits=5, shuffle=True, model=None, epochs=50, batch_size=180, verbose=0):
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=42)
        """
        # funcion para poder hacer validacion cruzada y ver 3 scores rapidamente r2, mae y rmse
    
        nota : el modo tensorflow no se ve muy confiable vamos a tener que evaluar esto
        """

        # Listas para almacenar los resultados de las métricas
        fold_mse = []
        fold_rmse = []
        fold_mae = []
        fold_r2 = []

        fold_mse_log = []
        fold_rmse_log = []
        fold_mae_log = []
        fold_r2_log = []

        for train_index, val_index in kf.split(X):
            # Seleccionar los datos para el fold actual
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            # Crear una nueva instancia del modelo para cada fold
            if model is None:
                raise ValueError("El parámetro 'model' no puede ser None. Debe ser una instancia de un modelo de scikit-learn o TensorFlow.")
            model_instance = model

            if tf_mode:
                # Configurar callbacks para TensorFlow
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=14, 
                    restore_best_weights=True
                )
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.2, 
                    patience=6, 
                    min_lr=1e-6
                )

                # Entrenar el modelo
                model_instance.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                                   epochs=epochs, batch_size=batch_size, verbose=verbose,
                                   callbacks=[early_stopping, reduce_lr])
            else:
                # Entrenar el modelo en modo scikit-learn o XGBoost
                model_instance.fit(X_train_fold, y_train_fold)

            # Realizar predicciones en el conjunto de validación
            y_pred_fold = model_instance.predict(X_val_fold)

            # Calcular las métricas para la partición actual
            mse = mean_squared_error(y_val_fold, y_pred_fold)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_fold, y_pred_fold)
            r2 = r2_score(y_val_fold, y_pred_fold)

            fold_mse.append(mse)
            fold_rmse.append(rmse)
            fold_mae.append(mae)
            fold_r2.append(r2)

            if log:
                # Revertir la transformación logarítmica
                predictions_valid_log = np.expm1(y_pred_fold)
                y_valid_n = np.expm1(y_val_fold)

                mse_log = mean_squared_error(y_valid_n, predictions_valid_log)
                rmse_log = np.sqrt(mse_log)
                mae_log = mean_absolute_error(y_valid_n, predictions_valid_log)
                r2_log = r2_score(y_valid_n, predictions_valid_log)

                fold_mse_log.append(mse_log)
                fold_rmse_log.append(rmse_log)
                fold_mae_log.append(mae_log)
                fold_r2_log.append(r2_log)
        
        if log == False:
            # Imprimir las métricas sin transformación matemática
            print("---------------------------------------------")    
            print("\nMétricas sin transformación matemática (exp)")
            print("---------------------------------------------")
            print(f"Mean across folds (MSE): {np.mean(fold_mse):.4f}")
            print(f"Standard Deviation of (MSE): {np.std(fold_mse):.4f}")
            print(f"\nMean across folds (RMSE): {np.mean(fold_rmse):.4f}")
            print(f"Standard Deviation of (RMSE): {np.std(fold_rmse):.4f}")
            print(f"\nMean across folds (MAE): {np.mean(fold_mae):.4f}")
            print(f"Standard Deviation of (MAE): {np.std(fold_mae):.4f}")
            print(f"\nMean across folds (R²): {np.mean(fold_r2):.4f}")
            print(f"Standard Deviation of (R²): {np.std(fold_r2):.4f}")

        if log == True:
            # Imprimir las métricas con transformación logarítmica
            print("\n---------------------------------------------")
            print("\nMétricas en términos reales de nuestra variable objetivo")
            print("---------------------------------------------")
            print(f"Mean across folds (MSE): {np.mean(fold_mse_log):.4f}")
            print(f"Standard Deviation of (MSE): {np.std(fold_mse_log):.4f}")
            print(f"\nMean across folds (RMSE): {np.mean(fold_rmse_log):.4f}")
            print(f"Standard Deviation of (RMSE): {np.std(fold_rmse_log):.4f}")
            print(f"\nMean across folds (MAE): {np.mean(fold_mae_log):.4f}")
            print(f"Standard Deviation of (MAE): {np.std(fold_mae_log):.4f}")
            print(f"\nMean across folds (R²): {np.mean(fold_r2_log):.4f}")
            print(f"Standard Deviation of (R²): {np.std(fold_r2_log):.4f}")
            
            
    def plot_error_distribution(self, y_true, y_pred):
        """
        Esta función toma los valores verdaderos y las predicciones, calcula el error,
        y muestra un gráfico con un histograma de errores y una curva KDE de la densidad.

        :param y_true: Array de valores verdaderos.
        :param y_pred: Array de valores predichos.
        """
        # Calcular errores
        errors = y_true - y_pred.flatten() # El método flatten() convierte un array multidimensional (por ejemplo, una matriz de 2D) en un array de una sola dimensión (un vector).

        #     # Crear la figura y el eje principal
        #     fig, ax1 = plt.subplots()

        # Crear la figura y el eje principal con un tamaño de figura mayor
        fig, ax1 = plt.subplots(figsize=(10, 6))  # Ajusta el tamaño aquí (ancho, alto)

        # Crear el histograma en el eje principal
        sns.histplot(errors, bins=50, edgecolor='k', alpha=0.7, ax=ax1)
        ax1.set_xlabel('Error = y_true - y_pred')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Errores de Predicción')

         # Ajustar el rango del eje x
        ax1.set_xlim([errors.min(), errors.max()])  # Ajusta según el rango de errores

        # Crear un segundo eje que comparte el mismo eje x
        ax2 = ax1.twinx()

        # Crear el KDE en el segundo eje
        sns.kdeplot(errors, ax=ax2, color='r')
        ax2.set_ylabel('Densidad')

        # Modificar la grilla del eje y de la derecha
        ax2.grid(True, which='both', axis='y', color='r', alpha=0.3)

        # Mostrar el gráfico
        plt.show()


    def eval_graf_errors(self, X_valid, y_valid, model_after_fit, log_transform_MODE = False):
        """
        Evalúa el rendimiento del modelo en el conjunto de validación y genera gráficos de errores.

        Parámetros:
        X_valid (pd.DataFrame): Conjunto de datos de validación (características).
        y_valid (pd.Series): Verdaderas etiquetas para el conjunto de validación.
        model_after_fit: Modelo entrenado para hacer predicciones.
        """
        # Verificar si el modelo está entrenado
        if not hasattr(model_after_fit, 'predict'):
            raise ValueError("El modelo debe estar entrenado y tener el método 'predict'.")

        # Hacer predicciones
        predictions_valid = model_after_fit.predict(X_valid)
        if log_transform_MODE:
            # Revertir la transformación logarítmica
            predictions_valid = np.expm1(predictions_valid)
            y_valid_n = np.expm1(y_valid)

        # Crear un DataFrame para seaborn
        results_df = pd.DataFrame({
            'Predicciones': predictions_valid.flatten(),
            'Reales': y_valid_n
        })

        # Crear el gráfico de regresión
        plt.figure(figsize=(10, 6))
        sns.regplot(x='Predicciones', y='Reales', data=results_df, scatter_kws={'s':10}, line_kws={'color':'red'})
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        plt.title('Gráfico de Regresión: Predicciones vs Valores Reales')
        plt.show()

        # Llamar a la función para graficar la distribución del error
        plot_error_distribution(y_valid_n, predictions_valid)
    