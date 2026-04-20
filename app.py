# app.py for froud detection 
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import joblib
import numpy as np
from datetime import datetime
from flask_cors import CORS
import sklearn

app = Flask(__name__)
CORS(app)

# Load dataset
pipe = joblib.load("fraud_pipeline.pkl")


#-----------
# Feature computation
#--------
@app.route('/fraud-detection', methods =["POST","GET"])
def fraud():
    return render_template('fraud-detection.html')
    
def compute_features_and_values(payload: dict):

	tx_type = payload.get("type", "PAYMENT").upper()
	amount = float(payload.get("amount",0.0))
	oldbalanceOrg = float(payload.get("oldbalanceOrg",0.0))
	newbalanceOrig = float(payload.get("newbalanceOrig",0.0))
	oldbalanceDest = float(payload.get("oldbalanceDest",0.0))
	newbalanceDest = float(payload.get("newbalanceDest",0.0))
	
	# Using server time
	hour = datetime.now().hour
	is_night = 1 if hour < 6 else 0
	amount_ratio = amount / (oldbalanceOrg + 1.0)
	
	sender_balance_change = oldbalanceOrg - newbalanceOrig
	receiver_balance_change = newbalanceDest - oldbalanceDest
	
	orig_balance_zero = 1 if oldbalanceOrg == 0 else 0
	dest_balance_zero = 1 if oldbalanceDest == 0 else 0
	type_TRANSFER = 1 if tx_type == "TRANSFER" else 0
	
	features = [
		amount,
		oldbalanceOrg,
		newbalanceOrig,
		oldbalanceDest,
		newbalanceDest,
		hour,
		is_night,
		amount_ratio,
		sender_balance_change,
		receiver_balance_change,
		orig_balance_zero,
		dest_balance_zero,
		type_TRANSFER
	]
	
	vals = {
		"tx_type": tx_type,
		"amount": amount,
		"oldbalanceOrg": oldbalanceOrg,
		"newbalanceOrig": newbalanceOrig,
		"oldbalanceDest": oldbalanceDest,
		"newbalanceDest": newbalanceDest,
		"hour": hour,
		"is_night": is_night,
		"amount_ratio": amount_ratio,
		"sender_balance_change": sender_balance_change,
		"receiver_balance_change": receiver_balance_change,
		"orig_balance_zero": orig_balance_zero,
		"dest_balance_zero": dest_balance_zero,
		"type_TRANSFER": type_TRANSFER
	}
	
	x = np.array(features).reshape(1, -1)
	return x, vals
	
	
#----------------------------------------
# Rule Based Froud Checks
#-----------------------------------------
def rule_based_checks(v):
    
    amount = v["amount"]
    sender_change = v["sender_balance_change"]
    receiver_change = v["receiver_balance_change"]
    oldbalanceOrg = v["oldbalanceOrg"]
    tx_type = v["tx_type"]
    
    # ✅ FIX 1
    if amount <= 0:
        return True, "Invalid transaction amount"
        
    #-----------------------------------------------------------
    # NEW RULE A: amount cannot exceed sender balance
    #-----------------------------------------------------------
    if amount > oldbalanceOrg:
        return True, "amount exceeds sender available balance"
        
    #-----------------------------------------------------------
    # NEW RULE B: receiver can not get more than amount
    #-----------------------------------------------------------
    if receiver_change > amount:
        return True, "receiver credited more than transferred amount"
        
    #------------------------------------------------------------
    # Rule 1: Sender loses = amount (+-20%)
    #------------------------------------------------------------
    if abs(sender_change - amount) > 0.20 * amount:
        return True, "sender balance change inconsistent with amount"
        
    #-------------------------------------------------------------
    # Rule 2 : Receiver gets enough credit (>=70%)
    #-------------------------------------------------------------
    if receiver_change < 0.70 * amount:
        return True, "receiver credited significantly less than expected"
        
    #--------------------------------------------------------------
    # Rule 3 : Receiver got zero in TRANSFER
    #--------------------------------------------------------------
    if tx_type == "TRANSFER" and receiver_change == 0:
        return True, "receiver balance not updated"
    
    #--------------------------------------------------------------
    # Rule 4 : total movement should be = 2 * amount (+-30%)
    #--------------------------------------------------------------
    total_movement = sender_change + receiver_change
    expected_total = 2 * amount
    
    if abs(total_movement - expected_total) > 0.30 * expected_total:
        return True, "inconsistent total money movement"
        
    # ✅ VERY IMPORTANT (ALWAYS RETURN)
    return False, ""
#---------------------------------------------------------------
# Routes
#---------------------------------------------------------------
@app.route('/')
def home():
	return render_template('index.html')
			
			
@app.route("/predict", methods =["POST"])
def predict():
	try:
		payload = request.get_json(force=True)
				
		x, vals = compute_features_and_values(payload)
				
		# Rule-based froud check
		flagged, reason = rule_based_checks(vals)
				
		if flagged:
			return jsonify({
				"is_fraud": 1,
				"probability": 1.0,
				"rule_flagged": True,
				"rule_reason": reason,
				"features": vals
			})
					
		# ML prediction
		pred = int(pipe.predict(x)[0])
		proba = float(pipe.predict_proba(x)[0][1])
				
		return jsonify({
				"is_fraud": pred,
				"probability": round(proba, 6),
				"rule_flagged": False,
				"rule_reason": "",
				"features": vals
		})
				
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
				
@app.route('/about')
def about():
    return render_template("aboutus.html")

@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        number = request.form.get('number')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Normally, you would process/store the form data here
        
        flash('Thank you for contacting us! We will get back to you shortly.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

app.secret_key = 'your_secret_key'  # Required for flash messages

# Dummy in-memory user store
users = {}


@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash('User already exists! Please login.', 'danger')
            return redirect(url_for('signin'))

        users[username] = password
        flash('Registration Successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signin.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users.get(username) == password:
            flash('Login Successful!', 'success')
            return "Welcome to Loan Approval Dashboard!"  # Placeholder
        else:
            flash('Invalid Credentials. Try Again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form.get("email")
        if email:  
            # Here you would normally send a reset link via email
            flash("A password reset link has been sent to your email.", "success")
            return redirect(url_for("forgot"))
        else:
            flash("Please enter a valid email address.", "danger")
    return render_template("forgot.html")				
#--------------------------------------------------------
# Start Server
#--------------------------------------------------------
if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=False)