# app.py - Dashboard lengkap seperti output Colab
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
import os
import webbrowser
import threading

app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    with open('static/stats.json', 'r') as f:
        stats = json.load(f)
    return jsonify(stats)

@app.route('/api/cluster_summary')
def get_cluster_summary():
    df = pd.read_csv('static/cluster_summary.csv')
    return jsonify(df.to_dict('records'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# FUNGSI UNTUK MEMBUKA BROWSER OTOMATIS
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Buat folder jika belum ada
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("="*50)
    print("🚀 DASHBOARD SIAP DIJALANKAN!")
    print("="*50)
    print("\n📍 Dashboard akan terbuka otomatis di browser")
    print("📍 Jika tidak terbuka, buka: http://localhost:5000")
    print("="*50)
    
    # Buka browser setelah 1.5 detik (server mulai)
    threading.Timer(1.5, open_browser).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)