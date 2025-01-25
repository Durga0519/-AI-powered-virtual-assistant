from flask import Flask, jsonify

app = Flask(__name__)

# Mock order data
orders = {
    "12345": {"status": "In transit", "delivery_date": "2025-01-25"},
    "67890": {"status": "Delivered", "delivery_date": "2025-01-20"}
}

@app.route('/order/<order_id>', methods=['GET'])
def track_order(order_id):
    """
    Return mock order status and delivery date based on order ID.
    """
    order = orders.get(order_id)
    if order:
        return jsonify(order)
    else:
        return jsonify({"error": "Order not found"}), 404

if __name__ == "__main__":
    app.run(port=5000)
