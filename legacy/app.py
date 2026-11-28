import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from flask import Flask, request, render_template_string, jsonify
import io, base64, csv, matplotlib.pyplot as plt, sqlite3

from scripts.backtest import load_data, run_backtest

app = Flask(__name__)

HTML_FORM = """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>ATLAS Back‑test UI</title>
  <style>
    body {font-family: Arial, sans-serif; margin: 2rem; background: #f5f5f5;}
    .container {background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    input[type='file'] {margin-bottom: 1rem;}
    label {display: block; margin-top: 0.5rem;}
    button {margin-top: 1rem; padding: 0.5rem 1rem;}
    table {border-collapse: collapse; width: 100%; margin-top: 1rem;}
    th, td {border: 1px solid #ddd; padding: 8px;}
    th {background: #f2f2f2;}
  </style>
</head>
<body>
  <div class='container'>
    <h1>ATLAS Back‑test</h1>
    <form action='/run' method='post' enctype='multipart/form-data'>
      <label>CSV file (hourly OHLC): <input type='file' name='csv' required></label>
      <label>EMA span: <input type='number' name='ema' value='20' min='1'></label>
      <label>ATR period: <input type='number' name='atr' value='14' min='1'></label>
      <label>Risk % per trade: <input type='number' step='0.1' name='risk' value='1.0'></label>
      <label>Stop‑loss multiplier: <input type='number' step='0.1' name='sl_mult' value='1.5'></label>
      <label>Enable short side: <input type='checkbox' name='short'></label>
      <button type='submit'>Run Back‑test</button>
    </form>
  </div>
</body>
</html>"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_FORM)

def dict_to_html_table(d):
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items())
    return f"<table>{rows}</table>"

def plot_price(df):
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df['close'], label='Close')
    plt.title('Price series')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

DB_PATH = os.path.join(os.path.dirname(__file__), 'results.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS backtest (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        final_equity REAL,
        net_profit REAL,
        trades INTEGER,
        win_rate REAL,
        max_drawdown REAL,
        avg_trade_h REAL,
        profit_factor REAL,
        avg_profit REAL
    )""")
    conn.commit()
    conn.close()

def save_result(results):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO backtest (timestamp, final_equity, net_profit, trades, win_rate, max_drawdown, avg_trade_h, profit_factor, avg_profit)
        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)""",
        (results['final_equity'], results['net_profit'], results['trades'],
         results['win_rate'], results['max_drawdown'], results['avg_trade_h'],
         results['profit_factor'], results['avg_profit']))
    conn.commit()
    conn.close()

# Ensure DB exists
init_db()

@app.route('/run', methods=['POST'])
def run():
    if 'csv' not in request.files:
        return jsonify({'error': 'CSV file missing'}), 400
    csv_file = request.files['csv']
    temp_path = Path('temp_upload.csv')
    csv_file.save(temp_path)
    df = load_data(temp_path)
    ema_span = int(request.form.get('ema', 20))
    atr_period = int(request.form.get('atr', 14))
    risk = float(request.form.get('risk', 1.0))
    sl_mult = float(request.form.get('sl_mult', 1.5))
    short = 'short' in request.form
    results = run_backtest(df, ema_span=ema_span, atr_period=atr_period,
                           risk_pct=risk, sl_mult=sl_mult, short=short)
    save_result(results)
    # CSV download link
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(results.keys())
    writer.writerow(results.values())
    csv_data = csv_buf.getvalue()
    csv_href = f"data:text/csv;charset=utf-8,{csv_data}"
    # Plot image
    img_b64 = plot_price(df)
    html = f"<h2>Back‑test results</h2>"
    html += dict_to_html_table(results)
    html += f"<p><a href='{csv_href}' download='backtest_summary.csv'>Download CSV</a></p>"
    html += f"<div><img src='data:image/png;base64,{img_b64}' alt='Price plot'></div>"
    try:
        os.remove(temp_path)
    except OSError:
        pass
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
