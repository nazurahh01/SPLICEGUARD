from flask import Flask, render_template, request, redirect, url_for, session, flash, abort, make_response, has_request_context
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ExifTags, UnidentifiedImageError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from captcha.image import ImageCaptcha
import numpy as np
import os, uuid, secrets, random, imghdr, re
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv
import mysql.connector
import smtplib
from email.mime.text import MIMEText
import math
import hashlib
import requests
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

# ================= LOAD ENV =================
load_dotenv()

# ================= APP CONFIG =================
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=15)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

THRESHOLD = 0.7
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
ALLOWED_IMGHDR = {"jpeg", "png"}
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_MINUTES = 1
OTP_EXPIRY_MINUTES = 5

# ================= LOAD MODEL =================

# Load the trained model (MobileNetV)
MODEL_PATH = "splicing_model_CASIA2_NEW.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train the model first.")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ================= USERS =================
USERS = {}

# ================= HISTORY =================
history = []

print("MYSQLHOST =", os.getenv("MYSQLHOST"))
print("MYSQLPORT =", os.getenv("MYSQLPORT"))
print("MYSQLDATABASE =", os.getenv("MYSQLDATABASE"))
print("MYSQLUSER =", os.getenv("MYSQLUSER"))

# ================= DATABASE =================

def get_db():
    conn = mysql.connector.connect(
        host=os.getenv("MYSQLHOST"),
        user=os.getenv("MYSQLUSER"),
        password=os.getenv("MYSQLPASSWORD"),
        database=os.getenv("MYSQLDATABASE"),
        port=int(os.getenv("MYSQLPORT")),
        autocommit=True,
        connection_timeout=30,
        ssl_disabled=False
    )

    print("CONNECTED =", conn.is_connected())

    return conn, conn.cursor(dictionary=True)

# ================= LOGIN ATTEMPTS =================
login_attempts = {}

def login_required(role=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Ensure the user is logged in (inside request context)
            if "username" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                abort(403)  # Forbidden if the role does not match
            return func(*args, **kwargs)
        return wrapper
    return decorator

def is_locked(username):
    info = login_attempts.get(username)
    if not info:
        return False
    locked_until = info.get("locked_until")
    if locked_until and datetime.now() < locked_until:
        return True
    if locked_until and datetime.now() >= locked_until:
        login_attempts[username] = {"count": 0, "locked_until": None}
        return False
    return False

def register_failed_attempt(username):
    info = login_attempts.get(username, {"count": 0, "locked_until": None})
    info["count"] += 1
    if info["count"] >= MAX_LOGIN_ATTEMPTS:
        info["locked_until"] = datetime.now() + timedelta(minutes=LOCKOUT_MINUTES)
    login_attempts[username] = info

def reset_login_attempts(username):
    login_attempts[username] = {"count": 0, "locked_until": None}

def generate_otp():
    return "".join(random.choices("0123456789", k=6))

def validate_password_strength(password):
    return (
        len(password) >= 12
        and re.search(r"[A-Z]", password)
        and re.search(r"[a-z]", password)
        and re.search(r"[0-9]", password)
        and re.search(r"[\W_]", password)
    )

# ================= ROOT =================
@app.route("/", methods=["GET", "POST"])
def index():

    if "username" in session:

        if session.get("role") == "admin":
            return redirect(url_for("admin_scan"))

        elif session.get("role") == "user":
            return redirect(url_for("tool"))
    else:
        # Non-logged-in user can only submit image
        result = None
        confidence = None
        image_path = None
        filename = None
        metadata = None
        error = None

        if request.method == "POST":

            guest_ip = request.remote_addr

            conn, cursor = get_db()
            print("CONN STATUS =", conn.is_connected())
            cursor.execute("""
                SELECT COUNT(*) AS total
                FROM guest_activity_log
                WHERE ip_address = %s
                AND DATE(timestamp) = CURDATE()
            """, (guest_ip,))

            guest_scan_count = cursor.fetchone()["total"]

            if guest_scan_count >= 15:
                error = "Guest scan limit reached. Please register to continue using SpliceGuard."
             
                return render_template(
                    "index.html",
                    result=result,
                    confidence=confidence,
                    image_path=image_path,
                    filename=filename,
                    metadata=metadata,
                    error=error
                )
            file = request.files.get("image")
            if not file or not file.filename:
                error = "Please select an image file."
            else:
                safe_name = secure_filename(file.filename)
                filename = f"{uuid.uuid4().hex}_{safe_name}"
                saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                try:
                    file.save(saved_path)
                    print("STEP 1 - file saved")
                    image = Image.open(saved_path).convert("RGB")
                    print("STEP 2 - image opened")
                    processed = preprocess_image(image)
                    print("STEP 3 - image preprocessed")
                    prediction = float(model.predict(processed, verbose=0)[0][0])
                    print(f"PREDICTION = {prediction}")
                    print("STEP 4 - prediction done")

                    if prediction >= THRESHOLD:
                        result = "Spliced (Fake)"
                        confidence = prediction * 100
                    else:
                        result = "Authentic (Original)"
                        confidence = (1 - prediction) * 100
                    
                    metadata = extract_metadata(saved_path)
                    image_path = "/" + saved_path.replace("\\", "/")

                    print("STEP 7")
                    add_guest_activity(
                        filename,
                        result,
                        round(confidence, 2)
                    )

                    print("STEP 8")
                except Exception as e:
                    error = f"Error processing the image: {str(e)}"

        return render_template("index.html", result=result, confidence=confidence, image_path=image_path, filename=filename, metadata=metadata, error=error)

# ================= LOGIN =================z
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":  # This ensures you're inside a request context
        if not validate_csrf():
            return render_template("login.html", error="Invalid security token. Please refresh and try again.")

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        captcha_input = request.form.get("captcha", "").strip().upper()

        if not username or not password or not captcha_input:
            generate_csrf_token()
            return render_template("login.html", error="All fields are required.")

        if is_locked(username):
            generate_csrf_token()
            return render_template("login.html", error="Too many failed attempts. Please try again later.", locked=True)

        session_captcha = session.get("captcha", "")
        session.pop("captcha", None)

        if not session_captcha or captcha_input != session_captcha:
            register_failed_attempt(username)
            generate_csrf_token()
            return render_template("login.html", error="Invalid CAPTCHA.")

        user = USERS.get(username)

        # If not in memory, check database
        if not user:
            user = get_user_by_username(username)

        if user and user.get("status") == "active" and check_password_hash(user["password_hash"], password):
            reset_login_attempts(username)

            session.clear()
            session.permanent = True
            session["username"] = user["username"]
            session["user_id"] = user["id"]

            from datetime import datetime

            subscription_active = False
            subscription_type = "FREE"

            if user.get("subscription_active") == 1:

                expiry = user.get("subscription_expiry")

                if expiry:

                    if isinstance(expiry, str):
                        expiry = datetime.strptime(
                            expiry,
                            "%Y-%m-%d %H:%M:%S"
                        )

                    if datetime.now() < expiry:

                        subscription_active = True
                        subscription_type = user.get(
                            "subscription_type",
                            "Premium"
                        )
                    else:
                        subscription_active = False
                        subscription_type = "FREE"

                        conn, cursor = get_db()
                        cursor.execute("""
                            UPDATE users
                            SET subscription_active = 0,
                                subscription_type = NULL,
                                subscription_expiry = NULL
                            WHERE id = %s
                        """, (user["id"],))

                        conn.commit()

                    subscription_active = True
                    subscription_type = user.get("subscription_type", "Premium")

            session["subscription_active"] = subscription_active
            session["subscription_type"] = subscription_type
            session["email"] = user["email"]
            session["role"] = user["role"]
            session["csrf_token"] = secrets.token_hex(16)

            if user["role"] != "admin":
                flash(
                    f"Welcome back, {user['username']}! You can now save reports and access case history.",
                    "success"
                )

            if user["role"] == "admin":
                admin_id = user.get("id")

                if admin_id:
                    add_admin_activity(admin_id, "login", "Admin logged in")

                return redirect(url_for("admin"))

            return redirect(url_for("tool"))

        register_failed_attempt(username)
        generate_csrf_token()
        flash(
            "Invalid username or password. Please try again.",
            "error"
        )
        return render_template("login.html")

    generate_csrf_token()
    return render_template("login.html")

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if not validate_csrf():
            flash("Invalid security token. Please refresh and try again.")
            return redirect(url_for("register"))

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required.")
            return redirect(url_for("register"))

        if username in USERS:
            flash("Username already exists.")
            return redirect(url_for("register"))

        existing_email_user = get_user_by_email(email)
        if existing_email_user:
            flash("Email already exists.")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        if not validate_password_strength(password):
            flash("Password must be at least 12 characters and include uppercase, lowercase, number, and symbol.")
            return redirect(url_for("register"))

        otp = generate_otp()

        session["otp_code"] = otp
        session["otp_created_at"] = datetime.now().isoformat()

        try:
            user_id = register_user_db(username, email, password)
            print(f"✅ User {username} saved to database with id {user_id}")
        except Exception as e:
            print(f"❌ Database error: {e}")
            flash("Database error. Please try again.")
            return redirect(url_for("register"))
        
        # Also save to in-memory dict for immediate access
        USERS[username] = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": "user",
            "status": "pending",
        }

        session["otp_email"] = email
        try:
            send_otp_email(email, otp)
            print(f"✅ Email sent to {email} with OTP: {otp}")
        except Exception as e:
            print(f"❌ Email failed: {e}")  # Log the error
        flash("OTP sent to your email. Please verify your account.")
        print(f"🔄 Redirecting to OTP page for email: {email}")
        return redirect(url_for("otp"))

    generate_csrf_token()
    return render_template("register.html")


# ================= OTP VERIFY FOR REGISTER =================
@app.route("/otp", methods=["GET", "POST"])
def otp():
    print(f"📍 OTP route accessed. Method: {request.method}, session otp_email: {session.get('otp_email')}")
    if "otp_email" not in session:
        print("❌ No otp_email in session, redirecting to register")
        return redirect(url_for("register"))

    user = get_user_by_email(session["otp_email"])
    if not user:
        print(f"❌ User not found for email: {session['otp_email']}")
        flash("User not found.")
        return redirect(url_for("register"))

    if request.method == "POST":
        if not validate_csrf():
            flash("Invalid security token. Please refresh and try again.")
            return redirect(url_for("otp"))

        otp_input = "".join(filter(str.isdigit, request.form.get("otp", "")))

        if otp_input != session.get("otp_code", ""):
            flash("Invalid OTP.")
            return redirect(url_for("otp"))

        otp_time = datetime.fromisoformat(session.get("otp_created_at"))
        if not otp_time or datetime.now() - otp_time > timedelta(minutes=OTP_EXPIRY_MINUTES):
            flash("OTP expired.")
            return redirect(url_for("otp"))

        user["status"] = "active"
        user["otp"] = None
        user["otp_created_at"] = None
        session.pop("otp_email", None)

        # Update database status
        try:
            conn, cursor = get_db()
            cursor.execute(
            "UPDATE users SET status='active' WHERE email=%s",
            (user["email"],)
        )
            conn.commit()
            print(f"✅ User {user['email']} activated in database")
        except Exception as e:
            print(f"❌ Database update error: {e}")

        flash("Account verified successfully. Please login.")
        return redirect(url_for("login"))

    generate_csrf_token()
    return render_template("otp.html")


# ================= RESEND OTP =================
@app.route("/resend_otp")
def resend_otp():
    if "otp_email" not in session:
        return redirect(url_for("register"))

    user = get_user_by_email(session["otp_email"])
    if not user:
        flash("User not found.")
        return redirect(url_for("register"))

    otp = generate_otp()
    user["otp"] = otp
    user["otp_created_at"] = datetime.now()

    try:
        send_otp_email(user["email"], otp)
    except Exception as e:
        print(f"Email failed: {e}")
    flash("New OTP sent.")
    return redirect(url_for("otp"))


# ================= FORGOT PASSWORD =================
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        if not validate_csrf():
            flash("Invalid security token. Please refresh and try again.")
            return redirect(url_for("forgot_password"))

        email = request.form.get("email", "").strip().lower()

        if not email:
            flash("Email is required.")
            return redirect(url_for("forgot_password"))

        user = get_user_by_email(email)

        if not user:
            flash("Email not found.")
            return redirect(url_for("forgot_password"))

        otp = generate_otp()
        session["reset_otp"] = otp
        session["reset_otp_created_at"] = datetime.now().isoformat()

        session["reset_email"] = email
        try:
            send_otp_email(email, otp)
        except Exception as e:
            print(f"Email failed: {e}")
        flash("OTP sent to your email.")
        return redirect(url_for("reset_password_otp"))

    generate_csrf_token()
    return render_template("forgot_password.html")


# ================= RESET PASSWORD OTP =================
@app.route("/reset-password-otp", methods=["GET", "POST"])
def reset_password_otp():
    if "reset_email" not in session:
        return redirect(url_for("forgot_password"))

    user = get_user_by_email(session["reset_email"])
    if not user:
        flash("User not found.")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        if not validate_csrf():
            flash("Invalid security token. Please refresh and try again.")
            return redirect(url_for("reset_password_otp"))

        otp_input = "".join(filter(str.isdigit, request.form.get("otp", "")))
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if otp_input != session.get("reset_otp", ""):
            flash("Invalid OTP.")
            return redirect(url_for("reset_password_otp"))

        otp_time = datetime.fromisoformat(
            session.get("reset_otp_created_at")
        )
        if not otp_time or datetime.now() - otp_time > timedelta(minutes=OTP_EXPIRY_MINUTES):
            flash("OTP expired.")
            return redirect(url_for("reset_password_otp"))

        if new_password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("reset_password_otp"))

        if not validate_password_strength(new_password):
            flash("Password must be at least 12 characters and include uppercase, lowercase, number, and symbol.")
            return redirect(url_for("reset_password_otp"))

        user["password_hash"] = generate_password_hash(new_password)
        session.pop("reset_email", None)

        # Update database
        conn, cursor = get_db()
        try:
            cursor.execute(
                "UPDATE users SET password_hash=%s WHERE email=%s",
                (generate_password_hash(new_password), user["email"]))
            conn.commit()

            session.pop("reset_email", None)
            session.pop("reset_otp", None)
            session.pop("reset_otp_created_at", None)
            print(f"✅ Password updated for {user['email']} in database")
        except Exception as e:
            print(f"❌ Database update error: {e}")

        flash("Password reset successful. Please login.")
        return redirect(url_for("login"))

    generate_csrf_token()
    return render_template("reset_otp.html")


# ================= TOOL =================
@app.route("/tool", methods=["GET", "POST"])
@login_required(role="user")
def tool():
    result = None
    confidence = None
    image_path = None
    filename = None
    metadata = None
    error = None

    if request.method == "POST":

    # ================= FREE USER LIMIT =================

        if not session.get("subscription_active"):

            conn, cursor = get_db()
            cursor.execute("""
                SELECT COUNT(*) AS total
                FROM history
                WHERE user_id = %s
                AND DATE(timestamp) = CURDATE()
            """, (session["user_id"],))

            user_scan_count = cursor.fetchone()["total"]

            if user_scan_count >= 10:

                error = "Free account daily scan limit reached. Upgrade your forensic workspace for unlimited access."

                return render_template(
                    "index.html",
                    result=result,
                    confidence=confidence,
                    image_path=image_path,
                    filename=filename,
                    metadata=metadata,
                    error=error
                )
            
        if not validate_csrf():
            return render_template("index.html", error="Invalid security token. Please refresh and try again.")

        file = request.files.get("image")

        if not file or not file.filename:
            return render_template("admin.html", error="Please select an image file.")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type. Only JPG, JPEG, and PNG are allowed.")

        safe_name = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{safe_name}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(saved_path)

            if not validate_real_image(saved_path):
                if os.path.exists(saved_path):
                    os.remove(saved_path)
                return render_template("index.html", error="Uploaded file is not a valid image.")

            image = Image.open(saved_path).convert("RGB")
            processed = preprocess_image(image)
            prediction = float(
                model.predict(processed, verbose=0)[0][0]
            )

            print(f"PREDICTION = {prediction}")

            if prediction >= THRESHOLD:
                result = "Spliced (Fake)"
                confidence = prediction * 100
            else:
                result = "Authentic (Original)"
                confidence = (1 - prediction) * 100

            metadata = extract_metadata(saved_path)

            # Get user ID from database
            user_record = get_user_by_username(session["username"])
            user_id = user_record["id"] if user_record else None

            # Save to database
            if user_id:
                add_history(user_id, filename, result, round(confidence, 2))

            # Also keep in memory for immediate display
            history.append({
                "filename": filename,
                "result": result,
                "confidence": round(confidence, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": session["username"]
            })

            image_path = "/" + saved_path.replace("\\", "/")

        except UnidentifiedImageError:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            error = "Unable to process the image."
        except Exception:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            error = "Unexpected error occurred during analysis."

    return render_template(
    "index.html",
    result=result,
    confidence=round(confidence, 2) if confidence else None,
    image_path=image_path,
    filename=filename,
    metadata=metadata,
    error=error
)

# ================= HISTORY =================
@app.route("/history")
@login_required()
def view_history():
    current_user = session["username"]

    conn, cursor = get_db()

    if session["role"] == "admin":
        # Get all history from database
        cursor.execute("""
            SELECT h.*, u.username as user
            FROM history h
            JOIN users u ON h.user_id = u.id
            ORDER BY h.timestamp DESC
        """)
        user_history = cursor.fetchall()
    else:
        # Get user's history from database
        cursor.execute("""
            SELECT h.*, u.username as user
            FROM history h
            JOIN users u ON h.user_id = u.id
            WHERE u.username = %s
            ORDER BY h.timestamp DESC
        """, (current_user,))
        user_history = cursor.fetchall()

    return render_template("history.html", history=user_history)

# ================= UPGRADE PLAN =================
@app.route("/upgrade")
@login_required()
def upgrade():
    return render_template("upgrade.html")

# ================= BUY PREMIUM PLAN =================
@app.route("/buy-premium")
@login_required()
def buy_premium():

    url = "https://dev.toyyibpay.com/index.php/api/createBill"

    payload = {

        "userSecretKey": "0nr9d5uk-b4h2-kjnz-ajwk-4vfdrisfcar4",

        "categoryCode": "kfn8u3x5",

        "billName": "SpliceGuard Premium",

        "billDescription": "Premium Monthly Subscription",

        "billPriceSetting": 1,

        "billPayorInfo": 1,

        "billAmount": 19000,

        "billReturnUrl": "https://spliceguard.site/payment-success",

        "billCallbackUrl": "https://spliceguard.site/payment-callback",

        "billExternalReferenceNo": str(session["user_id"]),

        "billTo": session["username"],

        "billEmail": session["email"],

        "billPhone": "0123456789"
    }

    response = requests.post(url, data=payload)

    result = response.json()

    bill_code = result[0]["BillCode"]

    payment_url = f"https://dev.toyyibpay.com/{bill_code}"

    return redirect(payment_url)

# ================= PAYMENT PLAN =================
@app.route("/payment-success")
@login_required()
def payment_success():

    expiry = datetime.now() + timedelta(days=30)

    conn, cursor = get_db()
    cursor.execute("""
        UPDATE users
        SET subscription_active = 1,
            subscription_type = 'Premium',
            subscription_expiry = %s
        WHERE id = %s
    """, (expiry, session["user_id"]))

    conn.commit()

    # UPDATE SESSION
    session["subscription_active"] = True
    session["subscription_type"] = "Premium"

    flash("Premium subscription activated successfully!", "success")

    return redirect(url_for("tool"))

# ================= ADMIN HISTORY =================
@app.route("/admin-history")
@login_required(role="admin")
def admin_history():

    current_user = session["username"]

    conn, cursor = get_db()
    cursor.execute("""
        SELECT h.*, u.username as user
        FROM history h
        JOIN users u ON h.user_id = u.id
        WHERE u.username = %s
        ORDER BY h.timestamp DESC
    """, (current_user,))

    admin_history_records = cursor.fetchall()

    return render_template(
        "history.html",
        history=admin_history_records
    )

# ================= REPORT =================
@app.route("/new-report/<filename>")
@login_required()
def generate_report(filename):
    current_user = session["username"]
    current_role = session["role"]

    conn, cursor = get_db()

    # ================= FREE REPORT LIMIT =================

    if current_role != "admin" and not session.get("subscription_active"):

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM report_logs
            WHERE user_id = %s
            AND DATE(created_at) = CURDATE()
        """, (session["user_id"],))

        report_count = cursor.fetchone()["total"]

        if report_count >= 5:

            # Allow report generation from History page
            if request.referrer and "/history" in request.referrer:
                pass

            else:
                flash(
                    "Daily report limit reached. Upgrade to Premium for unlimited forensic reports.",
                    "error"
                )

                return redirect(request.referrer or url_for("tool"))

    # Get record from database
    if current_role == "admin":
        cursor.execute("""
            SELECT h.*, u.username as user
            FROM history h
            JOIN users u ON h.user_id = u.id
            WHERE h.filename = %s
        """, (filename,))
    else:
        cursor.execute("""
            SELECT h.*, u.username as user
            FROM history h
            JOIN users u ON h.user_id = u.id
            WHERE h.filename = %s AND u.username = %s
        """, (filename, current_user))

    record = cursor.fetchone()

    if not record:
        return "Report not found", 404

    if current_role != "admin" and record["user"] != current_user:
        abort(403)

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(image_path):
        return "Image file not found", 404

    metadata = extract_metadata(image_path)
    print("DEBUG REPORT METADATA:", metadata)

    # Save report generation log
    if current_role != "admin":

        cursor.execute("""
            INSERT INTO report_logs (user_id)
            VALUES (%s)
        """, (session["user_id"],))

        conn.commit()
        cursor.close()
        conn.close()

    return render_template(
        "report.html",
        record=record,
        metadata=metadata,
        generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# ================= ADMIN =================
@app.route("/admin")
@login_required(role="admin")
def admin():
    # Get stats from database
    conn, cursor = get_db()
    cursor.execute("SELECT COUNT(*) as total_scans FROM history")
    total_scans = cursor.fetchone()["total_scans"]

    cursor.execute("SELECT COUNT(*) as total_forgery FROM history WHERE result LIKE '%Spliced%'")
    total_forgery = cursor.fetchone()["total_forgery"]

    cursor.execute("SELECT COUNT(DISTINCT user_id) as active_users FROM history")
    active_users = cursor.fetchone()["active_users"]

    # ================= GUEST USERS TODAY =================
    cursor.execute("""
        SELECT COUNT(DISTINCT ip_address) AS guest_today
        FROM guest_activity_log
        WHERE DATE(timestamp) = CURDATE()
    """)

    guest_today = cursor.fetchone()["guest_today"]

# ================= TOTAL GUEST ANALYSES =================
    cursor.execute("""
        SELECT COUNT(*) AS total_guest_uploads
        FROM guest_activity_log
    """)

    guest_uploads = cursor.fetchone()["total_guest_uploads"]

    current_user = session["username"]
    admin_id = get_user_by_username(current_user).get("id") if current_user else None
    if admin_id:
        add_admin_activity(admin_id, "view_dashboard", "Admin opened dashboard")

    # ================= SEARCH FILTER =================
    search = request.args.get("search", "")
    date = request.args.get("date", "")

    query = """
    SELECT *
    FROM (

        SELECT
            h.id,
            h.timestamp,
            u.username AS user,
            h.filename,
            h.result,
            h.confidence
        FROM history h
        JOIN users u ON h.user_id = u.id

        UNION ALL

        SELECT
            NULL AS id,
            g.timestamp,
            CONCAT('Guest (', g.ip_address, ')') AS user,
            g.filename,
            g.result,
            g.confidence
        FROM guest_activity_log g

    ) AS combined_logs
    WHERE 1=1
    """

    params = []

    # Search username
    if search:
        query += " AND user LIKE %s"
        params.append(f"%{search}%")

    # Filter by date
    if date:
        query += " AND DATE(timestamp) = %s"
        params.append(date)

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, params)

    history_records = cursor.fetchall()

    return render_template(
        "admin.html",
        history=history_records,
        total_scans=total_scans,
        total_forgery=total_forgery,
        active_users=active_users,
        guest_today=guest_today,
        guest_uploads=guest_uploads,
        search=search,
        date=date
    )

# ================= ADMIN SCAN =================
@app.route("/admin-scan", methods=["GET", "POST"])
@login_required(role="admin")
def admin_scan():

    result = None
    confidence = None
    image_path = None
    filename = None
    metadata = None
    error = None

    if request.method == "POST":

        file = request.files.get("image")

        if not file or not file.filename:
            return render_template(
                "admin.html",
                error="Please select an image file."
            )

        safe_name = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex}_{safe_name}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(saved_path)

            image = Image.open(saved_path).convert("RGB")
            processed = preprocess_image(image)

            prediction = float(
                model.predict(processed, verbose=0)[0][0]
            )

            if prediction >= THRESHOLD:
                result = "Spliced (Fake)"
                confidence = prediction * 100
            else:
                result = "Authentic (Original)"
                confidence = (1 - prediction) * 100

            metadata = extract_metadata(saved_path)

            admin_record = get_user_by_username(session["username"])
            admin_id = admin_record["id"]

            add_history(
                admin_id,
                filename,
                result,
                round(confidence, 2)
            )

            image_path = "/" + saved_path.replace("\\", "/")

        except Exception:
            error = "Unexpected error occurred during analysis."

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence, 2) if confidence else None,
        image_path=image_path,
        filename=filename,
        metadata=metadata,
        error=error
    )
# ================= admin report =================

@app.route("/admin-report/<int:history_id>")
@login_required(role="admin")
def admin_report(history_id):

    conn, cursor = get_db()

    cursor.execute("""
        SELECT h.*, u.username
        FROM history h
        JOIN users u ON h.user_id = u.id
        WHERE h.id = %s
    """, (history_id,))

    record = cursor.fetchone()

    if not record:
        return "Record not found", 404

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        record["filename"]
    )

    metadata = extract_metadata(image_path)

    return render_template(
        "admin_report.html",
        record=record,
        metadata=metadata,
        generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_real_image(filepath):
    detected = imghdr.what(filepath)
    return detected in ALLOWED_IMGHDR

def generate_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return session["csrf_token"]

@app.context_processor
def inject_csrf():
    if has_request_context() and "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return {"csrf_token": session.get("csrf_token", "") if has_request_context() else ""}

def validate_csrf():
    form_token = request.form.get("csrf_token", "")
    session_token = session.get("csrf_token", "")
    return bool(form_token and session_token and secrets.compare_digest(form_token, session_token))

# ================= DATABASE HELPERS =================
def get_user_by_username(username):
    user = USERS.get(username)
    if user:
        if user.get("id"):
            return user
        # If the user exists in memory but was created in the DB, fetch the DB record.
        conn, cursor = get_db()
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        db_user = cursor.fetchone()
        return db_user or user
    conn, cursor = get_db()
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    return cursor.fetchone()

def get_user_by_email(email):

    for user in USERS.values():
        if user.get("email", "").lower() == email.lower():
            if user.get("id"):
                return user

            conn, cursor = get_db()
            cursor.execute(
                "SELECT * FROM users WHERE email=%s",
                (email,)
            )

            db_user = cursor.fetchone()

            cursor.close()
            conn.close()

            return db_user or user

    conn, cursor = get_db()

    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (email,)
    )

    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result

def register_user_db(username, email, password):
    password_hash = generate_password_hash(password)
    conn, cursor = get_db()
    cursor.execute("""
        INSERT INTO users (
            username, email, password_hash,
            role, status
        )
        VALUES (%s, %s, %s, %s, %s)""", (
        username,
        email,
        password_hash,
        'user',
        'pending'
    ))
    conn.commit()
    return cursor.lastrowid

def add_history(user_id, filename, result, confidence):
    conn, cursor = get_db()
    cursor.execute("""
        INSERT INTO history (user_id, filename, result, confidence)
        VALUES (%s, %s, %s, %s)
    """, (user_id, filename, result, confidence))
    conn.commit()

# ================= GUEST ACTIVITY =================
def add_guest_activity(filename, result, confidence):

    conn, cursor = get_db()
    ip_address = request.remote_addr

    cursor.execute("""
        INSERT INTO guest_activity_log
        (filename, result, confidence, ip_address)
        VALUES (%s, %s, %s, %s)
    """, (
        filename,
        result,
        confidence,
        ip_address
    ))

    conn.commit()

def add_admin_activity(admin_user_id, action, details=None):
    conn, cursor = get_db()
    cursor.execute("""
        INSERT INTO admin_activity_log (admin_user_id, action, details)
        VALUES (%s, %s, %s)
    """, (admin_user_id, action, details))
    conn.commit()

def get_user_history(user_id):
    conn, cursor = get_db()
    cursor.execute("SELECT * FROM history WHERE user_id=%s ORDER BY timestamp DESC", (user_id,))
    return cursor.fetchall()

# ================= OTP EMAIL =================
def send_otp_email(to_email, otp):
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    if not smtp_user or not smtp_pass:
        raise ValueError("SMTP credentials not found in .env file")
    msg = MIMEText(f"Your OTP code is: {otp}\nValid for 5 minutes.")
    msg["Subject"] = "SpliceGuard OTP Verification"
    msg["From"] = smtp_user
    msg["To"] = to_email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

# ================= IMAGE HELPERS =================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

def clean_metadata_value(value):
    if value is None or value == "":
        return "Not Available"

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return "Unreadable"

    text = str(value).replace("\x00", "").strip()

    if not text:
        return "Not Available"

    return text


def calculate_file_hash(image_path):
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for block in iter(lambda: file.read(4096), b""):
            sha256.update(block)

    return sha256.hexdigest()


def extract_metadata(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
        file_size_kb = round(os.path.getsize(image_path) / 1024, 2)

        if width > 0 and height > 0:
            gcd_value = math.gcd(width, height)
            aspect_ratio = f"{width // gcd_value}:{height // gcd_value}"
            megapixels = round((width * height) / 1_000_000, 2)
        else:
            aspect_ratio = "Not Available"
            megapixels = "Not Available"

        metadata = {
            "filename": os.path.basename(image_path),
            "format": clean_metadata_value(image.format),
            "resolution": f"{width} x {height}",
            "aspect_ratio": aspect_ratio,
            "megapixels": megapixels,
            "file_size_kb": file_size_kb,
            "camera_model": "Not Available",
            "orientation": "Not Available",
            "software": "Not Available",
            "datetime": "Not Available",
            "exif_status": "Not Available",
            "file_hash": calculate_file_hash(image_path)
        }

        orientation_map = {
            1: "Normal",
            2: "Mirrored horizontal",
            3: "Rotated 180°",
            4: "Mirrored vertical",
            5: "Mirrored horizontal and rotated 270°",
            6: "Rotated 90° clockwise",
            7: "Mirrored horizontal and rotated 90°",
            8: "Rotated 90° counter-clockwise"
        }

        try:
            exif = image._getexif()

            if exif:
                metadata["exif_status"] = "Available"

                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)

                    if tag_name == "Model":
                        metadata["camera_model"] = clean_metadata_value(value)

                    elif tag_name == "Orientation":
                        metadata["orientation"] = orientation_map.get(
                            value,
                            clean_metadata_value(value)
                        )

                    elif tag_name == "Software":
                        metadata["software"] = clean_metadata_value(value)

                    elif tag_name in ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]:
                        if metadata["datetime"] == "Not Available":
                            metadata["datetime"] = clean_metadata_value(value)

        except Exception:
            metadata["exif_status"] = "Unreadable"

        return metadata

# ================= CAPTCHA =================
@app.route("/captcha")
def captcha():
    image = ImageCaptcha()
    captcha_text = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))
    session["captcha"] = captcha_text
    data = image.generate(captcha_text)
    response = make_response(data.read())
    response.headers["Content-Type"] = "image/png"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response

# ================= LOGOUT =================
@app.route("/logout")
@login_required()
def logout():
    session.clear()
    return redirect(url_for("login"))

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)