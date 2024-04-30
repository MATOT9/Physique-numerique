import heapq
import itertools
import multiprocessing
from typing import Any, Iterator
from collections import Counter, deque

import numpy as np
from numpy.typing import NDArray


CPU_COUNT = multiprocessing.cpu_count()


class HuffmanNode:
    """Noeud utilisé pour la construction de l'arbre de Huffman."""

    __slots__ = "char", "freq", "left", "right"

    def __init__(self, char: Any, freq: int):
        """Initialise un nouveau noeud de l'arbre de Huffman.

        Paramètres
        ----------
            char
                Le caractère ou la valeur stocké par le noeud, `None` pour les
                noeuds internes.
            freq
                La fréquence d'apparition du caractère dans le corpus analysé.

        """
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        """Comparaison pour le maintien de l'ordre dans la file de priorité."""
        return self.freq < other.freq


def build_huffman_tree(freqs: dict[Any, int]) -> HuffmanNode:
    """Construit l'arbre de Huffman à partir d'un dictionnaire de fréquences.

    Paramètres
    ----------
        freqs
            Un dictionnaire où les clés sont des éléments (caractères, etc.) et
            les valeurs sont les fréquences.

    Retourne
    --------
        HuffmanNode
            Le noeud racine de l'arbre de Huffman.

    """
    # Remplir la file de priorité avec les noeuds initiaux.
    priority_queue = list(itertools.starmap(HuffmanNode, freqs.items()))
    heapq.heapify(priority_queue)

    # Fusionner les noeuds jusqu'à ce qu'il ne reste qu'un seul arbre.
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]


def generate_codes(
    node: HuffmanNode,
    prefix: str = "",
    code_book: dict[Any, str] = {},
) -> dict[Any, str]:
    """Génère un dictionnaire de codes Huffman à partir de l'arbre de Huffman.

    Paramètres
    ----------
        node
            Le noeud actuel de l'arbre.
        prefix
            Le préfixe de code actuel.
        code_book : dict, optional
            Le livre de codes accumulé.

    Retourne
    --------
        code_book
            Un dictionnaire avec des éléments comme clés et des codes
            binaires comme valeurs.

    """
    if node:
        if node.char is not None:
            code_book[node.char] = prefix
        generate_codes(node.left, prefix + "0", code_book)
        generate_codes(node.right, prefix + "1", code_book)
    return code_book


def huffman_encode(data):
    """Encode les données via la méthode de Huffman.

    Paramètres
    ----------
        data
            Les données à encoder.

    Retourne
    --------
        encoded_string
            La chaîne de caractères encodée.
        root
            Le noeud à la racine de l'arbre de Huffman.

    """
    # Compte les occurences des données
    if isinstance(data[0], Iterator):
        data1, data2 = data
    else:
        data1 = data2 = data

    freqs = Counter(data1)
    root = build_huffman_tree(freqs)
    codes = generate_codes(root)

    # Encode avec le dictionnaire de Huffman généré
    encoded_string = "".join(map(codes.get, data2))
    return encoded_string, root


def huffman_decode(
    encoded_data: str,
    root: HuffmanNode,
) -> Any:
    """Décode les données encodées avec le code de Huffman.

    Paramètres
    ----------
        encoded_data
            Les données encodées à déchiffrer.
        root
            Le noeud racine de l'arbre de Huffman.
        shape
            La forme originale des données si elles doivent être
            reconstituées dans un array numpy.

    Retourne
    --------
        decoded
            Les données décodées, sous forme de chaîne ou d'array numpy.

    """
    # Use deque for efficient append operations
    decoded = deque()
    node = root
    for bit in encoded_data:
        # Traverse the Huffman tree based on the current bit
        node = node.left if bit == "0" else node.right

        if node.char is not None:
            # The node is a leaf node
            decoded.append(node.char)
            node = root

    return decoded


def rle_encode(data: Any) -> tuple[Any, NDArray]:
    """Encode ou inverse l'encodage RLE des données texte ou array NumPy.

    Paramètres
    ----------
        data
            Les données à encoder ou à décoder.
        shape
            La forme du tableau numpy lors de la décompression.
        reverse
            Indique si l'encodage doit être inversé.

    Retourne
    --------
        result
            Les données encodées en RLE ou décodées si `reverse` est True.

    Soulève
    -------
        TypeError
            Si le type de données spécifié n'est pas pris en charge.

    """
    if isinstance(data, str):
        if not data:
            return []

        result = deque()
        last_char = data[0]
        count = 1
        for char in data[1:]:
            if isinstance(data, str):
                if char == last_char:
                    count += 1
                else:
                    result.append((last_char, count))
                    last_char = char
                    count = 1

            result.append((last_char, count))
            return result

    elif isinstance(data, np.ndarray):
        if data.size == 0:
            return []

        # Aplatit pour traitement 1D et trouve les changements de valeur
        data = data.ravel()
        positions = np.where(np.diff(data) != 0)[0] + 1

        # Ajoute les positions de début et de fin et récupère les valeurs
        positions = np.r_[0, positions, len(data)]
        values = data[positions[:-1]]

        # Calcule la longueur de chaque segment et crée une table
        counts = np.diff(positions)
        table = itertools.tee(zip(values, counts), 2)
        return table

    else:
        raise TypeError("Type de données non supporté pour l'encodage RLE")


def rle_decode(encoding, odata):
    """Inverse l'encodage RLE des données texte ou array NumPy.

    Paramètres
    ----------
        encoding
            Les données à à décoder.
        odata
            Less données avant la compression. Permet de reconstituées
            la forme de l'array NumPy initiale.

    Retourne
    --------
        decoded
            Les données déencodées en RLE.

    """
    def _rle_decode(values: Any, counts: np.ndarray) -> Any:
        """Fonction auxiliaire pour décompresser chaque segment."""
        if isinstance(values, str):
            decoded = "".join(values * counts)
        else:
            decoded = np.repeat(values, counts)
        return decoded

    # Application de la décompression à tous les segments
    decoded = itertools.starmap(_rle_decode, encoding)
    if isinstance(odata, str):
        decoded = "".join(decoded)
    elif isinstance(odata, np.ndarray):
        decoded = np.concatenate([*decoded]).reshape(odata.shape)
    return decoded


def process_data(data: Any, verbose: bool = True) -> tuple[Any]:
    """Encodage et décodage d'un bloc de données en utilisant RLE et Huffman.

    Cette fonction prend en entrée un canal de l'image, le compresse en utilisant
    l'encodage Run-Length (RLE) suivi de l'encodage Huffman, puis décode les données
    pour vérifier l'intégrité et la réversibilité des encodages appliqués.

    Paramètres
    ----------
        data
            Bloc de données à traiter.
        verbose
            Si vrai, retourne également la taille des données après la compression.

    Retourne
    -------
        encoded_data
            Les données encodées en Huffman.
        decoded_data
            Les données décodées après avoir appliqué Huffman et RLE pour revenir
            à l'état original du canal.
        csize
            La taille compressée du bloc de données.

    """
    encoded_text = rle_encode(data)
    encoded_data = huffman_encode(encoded_text)
    decoded_data = huffman_decode(*encoded_data)
    decoded_data = rle_decode(decoded_data, data)
    csize = len(encoded_data[0]) if verbose is True else None
    return encoded_data, decoded_data, csize


def parallel_huffman(
    blocks: NDArray,
    verbose: bool = True,
) -> tuple[NDArray]:
    """Exécution parallèle des encodages RLE et Huffman les canaux d'une image.

    Utilise le multiprocessing pour traiter chaque canal de l'image en parallèle,
    ce qui permet d'améliorer la performance sur des machines multicœurs. La fonction
    s'adapte aussi bien aux images en niveaux de gris (2D) qu'aux images couleur
    (avec plusieurs canaux).

    Paramètres
    ----------
        blocks
            Les données de l'image, pré-divisées en blocs ou en canaux.
        verbose
            Si vrai, retourne également la taille des données en MB après
            la compression.

    Retourne
    --------
        encoded
            Les données encodées de chaque canal.
        decoded
            Les données décodées de chaque canal, permettant de vérifier
            l'intégrité.

    """
    print(f"Compression des données avec {CPU_COUNT} cœurs CPU...\n")
    with multiprocessing.Pool(CPU_COUNT) as pool:
        results = pool.map(process_data, blocks)

    # Séparation des résultats encodés et décodés et conversion en array
    encoded, decoded, csizes = zip(*results)
    encoded, decoded = np.asarray(encoded), np.asarray(decoded)
    csize = np.asarray(csizes).sum() * 1e-6 if verbose else None
    return encoded, decoded, csize
