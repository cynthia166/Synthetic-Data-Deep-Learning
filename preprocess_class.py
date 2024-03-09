class PreprocessInput:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # Aquí puedes agregar la lógica para preprocesar los datos
        # Puedes acceder a los datos utilizando self.data

        # Ejemplo de preprocesamiento: convertir los datos a minúsculas
        preprocessed_data = self.data.lower()

        return preprocessed_data

    def additional_method(self):
        # Aquí puedes agregar la lógica para el nuevo método
        # Puedes acceder a los datos utilizando self.data o cualquier otro atributo de la clase
        pass


    
class MiClase:
    def __init__(self):
        # Inicialización de atributos que siempre son necesarios
        self.atributo_general = None

    def metodo_especifico_1(self, input_1):
        # Operación específica 1
        print(f"Operación específica 1 con {input_1}")

    def metodo_especifico_2(self, input_2):
        # Operación específica 2
        print(f"Operación específica 2 con {input_2}")

    def ejecutar_flujo(self, usar_metodo_1, input_1=None, usar_metodo_2=False, input_2=None):
        """
        Método gestor que controla la ejecución de otros métodos según las necesidades.
        
        :param usar_metodo_1: Booleano que indica si se debe llamar a metodo_especifico_1.
        :param input_1: Entrada para metodo_especifico_1 si se utiliza.
        :param usar_metodo_2: Booleano que indica si se debe llamar a metodo_especifico_2.
        :param input_2: Entrada para metodo_especifico_2 si se utiliza.
        """
        if usar_metodo_1 and input_1 is not None:
            self.metodo_especifico_1(input_1)
        
        if usar_metodo_2 and input_2 is not None:
            self.metodo_especifico_2(input_2)

# Ejemplo de uso
mi_objeto = MiClase()
mi_objeto.ejecutar_flujo(usar_metodo_1=True, input_1='Dato para método 1', usar_metodo_2=True, input_2='Dato para método 2')
