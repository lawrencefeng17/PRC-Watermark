"""
Implementation of Huffman encoding.
"""

def build_huffman_tree(frequencies):
    #using a dictionary instead of a heap
    nodes = {symbol: {"freq": freq, "left": None, "right": None, "symbol": symbol} for symbol, freq in frequencies.items()}
    while len(nodes) > 1:
        #find two least frequent nodes
        left_symbol, right_symbol = sorted(nodes, key=lambda symbol: nodes[symbol]["freq"])[:2]
        left_node = nodes.pop(left_symbol)
        right_node = nodes.pop(right_symbol)
        new_node = {"freq": left_node["freq"] + right_node["freq"], "left": left_node, "right": right_node}
        nodes[f"{left_symbol}, {right_symbol}"] = new_node

    [(root_symbol, root_node)] = nodes.items()
    return root_node

def generate_huffman_codes(tree):
    codes = {}
    def traverse(node, current_code=""):
        if node["left"] is None and node["right"] is None:
            # Store the code for the symbol, not the frequency
            codes[node["symbol"]] = current_code
            return
        traverse(node["left"], current_code + "0")
        traverse(node["right"], current_code + "1")
    traverse(tree)
    return codes

def huffman_encode(frequencies):
  tree = build_huffman_tree(frequencies)
  codes = generate_huffman_codes(tree)
  encoding = {token: codes[token] for token, freq in frequencies.items()}
  return encoding

def huffman_decode(encoding, encoded_string):
    decoding = {code: symbol for symbol, code in encoding.items()} # This line was correct
    decoded_sequence = []
    current_code = ""
    for bit in encoded_string:
        current_code += bit
        if current_code in decoding:
            decoded_sequence.append(decoding[current_code])
            current_code = ""
    return decoded_sequence
