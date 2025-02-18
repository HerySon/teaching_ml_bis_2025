import os
import shutil

def clean_directory(directory):
    """Nettoie les fichiers inutiles du répertoire."""
    patterns_to_remove = [
        '__pycache__',
        '.ipynb_checkpoints',
        '.DS_Store',
        '.coverage',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.Python',
        'build',
        'develop-eggs',
        'dist',
        'downloads',
        'eggs',
        '.eggs',
        'lib',
        'lib64',
        'parts',
        'sdist',
        'var',
        'wheels',
        '*.egg-info',
        '.installed.cfg',
        '*.egg'
    ]
    
    for root, dirs, files in os.walk(directory):
        # Supprime les dossiers correspondant aux patterns
        for pattern in patterns_to_remove:
            if pattern in dirs:
                shutil.rmtree(os.path.join(root, pattern))
                print(f"Supprimé: {os.path.join(root, pattern)}")
        
        # Supprime les fichiers correspondant aux patterns
        for pattern in patterns_to_remove:
            if '*' in pattern:
                extension = pattern.replace('*', '')
                for file in files:
                    if file.endswith(extension):
                        os.remove(os.path.join(root, file))
                        print(f"Supprimé: {os.path.join(root, file)}")

if __name__ == "__main__":
    clean_directory('teaching_ml_bis_2025')