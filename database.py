
import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH

def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table (name field as per requirements, but we use username for login)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT,  -- Additional name field as per requirements
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            height REAL,
            weight REAL,
            age INTEGER,
            activity_level TEXT,
            preferred_language TEXT DEFAULT 'en',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Image uploads table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            kl_grade INTEGER NOT NULL,
            confidence REAL NOT NULL,
            gradcam_path TEXT,
            model_version TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES image_uploads (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Radiology reports table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS radiology_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            report_text_en TEXT NOT NULL,
            report_text_translated TEXT,
            language TEXT DEFAULT 'en',
            pdf_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Prescriptive reports table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prescriptive_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            report_text TEXT NOT NULL,
            pdf_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def add_user(username, email, password_hash, name=None):
    """Add a new user to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, name, email, password_hash) VALUES (?, ?, ?, ?)',
            (username, name or username, email, password_hash)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_username(username):
    """Get user by username"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def update_user_profile(user_id, height=None, weight=None, age=None, activity_level=None, preferred_language=None):
    """Update or create user profile"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if profile exists
    cursor.execute('SELECT id FROM user_profiles WHERE user_id = ?', (user_id,))
    profile = cursor.fetchone()
    
    if profile:
        # Update existing profile
        updates = []
        values = []
        if height is not None:
            updates.append('height = ?')
            values.append(height)
        if weight is not None:
            updates.append('weight = ?')
            values.append(weight)
        if age is not None:
            updates.append('age = ?')
            values.append(age)
        if activity_level is not None:
            updates.append('activity_level = ?')
            values.append(activity_level)
        if preferred_language is not None:
            updates.append('preferred_language = ?')
            values.append(preferred_language)
        
        if updates:
            values.append(user_id)
            cursor.execute(
                f'UPDATE user_profiles SET {", ".join(updates)} WHERE user_id = ?',
                values
            )
    else:
        # Create new profile
        cursor.execute(
            '''INSERT INTO user_profiles (user_id, height, weight, age, activity_level, preferred_language)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (user_id, height, weight, age, activity_level, preferred_language or 'en')
        )
    
    conn.commit()
    conn.close()

def get_user_profile(user_id):
    """Get user profile"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

def save_image_upload(user_id, filename, filepath):
    """Save image upload record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO image_uploads (user_id, filename, filepath) VALUES (?, ?, ?)',
        (user_id, filename, filepath)
    )
    image_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return image_id

def save_prediction(image_id, user_id, kl_grade, confidence, gradcam_path, model_version):
    """Save prediction results"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO predictions (image_id, user_id, kl_grade, confidence, gradcam_path, model_version)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (image_id, user_id, kl_grade, confidence, gradcam_path, model_version)
    )
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def save_radiology_report(prediction_id, user_id, report_text_en, report_text_translated, language, pdf_path):
    """Save radiology report"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO radiology_reports (prediction_id, user_id, report_text_en, report_text_translated, language, pdf_path)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (prediction_id, user_id, report_text_en, report_text_translated, language, pdf_path)
    )
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id

def save_prescriptive_report(prediction_id, user_id, report_text, pdf_path):
    """Save prescriptive report"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO prescriptive_reports (prediction_id, user_id, report_text, pdf_path) VALUES (?, ?, ?, ?)',
        (prediction_id, user_id, report_text, pdf_path)
    )
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id

def get_user_predictions(user_id, limit=10):
    """Get recent predictions for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT p.*, i.filename, i.upload_date
        FROM predictions p
        JOIN image_uploads i ON p.image_id = i.id
        WHERE p.user_id = ?
        ORDER BY p.prediction_date DESC
        LIMIT ?
    ''', (user_id, limit))
    predictions = cursor.fetchall()
    conn.close()
    return predictions

def delete_prediction(prediction_id, user_id):
    """
    Delete a prediction and all related data (reports, files)
    Returns True if successful, False otherwise
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # First, verify the prediction belongs to the user
        cursor.execute('SELECT id, gradcam_path FROM predictions WHERE id = ? AND user_id = ?', 
                      (prediction_id, user_id))
        prediction = cursor.fetchone()
        
        if not prediction:
            conn.close()
            return False
        
        # Get related data before deletion
        cursor.execute('SELECT pdf_path FROM radiology_reports WHERE prediction_id = ?', (prediction_id,))
        radiology_reports = cursor.fetchall()
        
        cursor.execute('SELECT pdf_path FROM prescriptive_reports WHERE prediction_id = ?', (prediction_id,))
        prescriptive_reports = cursor.fetchall()
        
        # Delete related reports (cascade delete)
        cursor.execute('DELETE FROM radiology_reports WHERE prediction_id = ?', (prediction_id,))
        cursor.execute('DELETE FROM prescriptive_reports WHERE prediction_id = ?', (prediction_id,))
        
        # Delete the prediction
        cursor.execute('DELETE FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id))
        
        conn.commit()
        conn.close()
        
        # Optionally delete files (Grad-CAM images, PDFs)
        # This is optional - you can keep files or delete them
        # For now, we'll just delete from database
        
        return True
    except Exception as e:
        print(f"Error deleting prediction: {e}")
        conn.rollback()
        conn.close()
        return False


