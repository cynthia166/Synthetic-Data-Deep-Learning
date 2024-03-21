import os
p = "/Users/cgarciay/Desktop/Laval_Master_Computer/research/Synthetic-Data-Deep-Learning"

def listar_directorios(ruta):
    for raiz, dirs, archivos in os.walk(ruta):
        nivel = raiz.replace(ruta, '').count(os.sep)
        indentacion = ' ' * 4 * (nivel)
        print('{}{}/'.format(indentacion, os.path.basename(raiz)))
        sub_indentacion = ' ' * 4 * (nivel + 1)
        for archivo in archivos:
            print('{}{}'.format(sub_indentacion, archivo))

# Reemplaza 'mi_ruta' con la ruta de la carpeta que quieres listar
listar_directorios(p)