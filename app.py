import json
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    send_file,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from predictor import RankingPredictor

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "products_with_images.csv"
USERS_PATH = BASE_DIR / "data" / "users.json"
CATALOG_SIZE = 192
app = Flask(__name__, template_folder="Templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "mr-demo-secret-key")


def ensure_data_files():
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text("{}", encoding="utf-8")


def load_users():
    ensure_data_files()
    with USERS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_users(users):
    ensure_data_files()
    with USERS_PATH.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


def login_required():
    return "user" in session


def current_user():
    if not login_required():
        return None
    return load_users().get(session["user"])


def load_predictor():
    return RankingPredictor(
        "saved_models/best_ml_model.pkl",
        "saved_models/best_dl_model.h5",
        "saved_models/preprocessor.pkl",
        "saved_models/model_metadata.json",
    )


def prepare_model_features(df):
    working = df.copy()
    working["price_per_rating"] = working["price"] / working["rating"].replace(0, np.nan)
    working["price_per_rating"] = working["price_per_rating"].replace([np.inf, -np.inf], np.nan).fillna(working["price"])
    working["review_log"] = np.log1p(working["review_count"].clip(lower=0))
    working["description_keyword_interaction"] = working["description_length"] * working["keyword_score"]
    working["popularity_content_interaction"] = working["popularity_score"] * working["content_score"]
    working["is_local_brand"] = (working["brand_type"].str.lower() == "local").astype(int)
    return working


def choose_catalog_slice(df):
    frames = []
    target_per_group = max(12, CATALOG_SIZE // 8)
    for _, group in df.groupby(["brand_type", "category"], sort=True):
        take = min(len(group), target_per_group)
        frames.append(group.sample(n=take, random_state=42) if len(group) > take else group)

    catalog = pd.concat(frames, ignore_index=True)
    if len(catalog) > CATALOG_SIZE:
        catalog = catalog.sample(n=CATALOG_SIZE, random_state=42)
    return catalog.sort_values("product_id").reset_index(drop=True)


def build_fair_ranking(scored):
    working = scored.copy()
    working["visibility_score"] = working["score"]
    working["visibility_score"] += np.where(working["brand_type"].str.lower() == "local", 0.12, 0.0)
    working["visibility_score"] += np.where(working["seller_type"].str.lower() == "small", 0.05, 0.0)

    local_pool = (
        working.loc[working["brand_type"].str.lower() == "local"]
        .sort_values(["visibility_score", "score", "rating"], ascending=[False, False, False])
        .to_dict(orient="records")
    )
    branded_pool = (
        working.loc[working["brand_type"].str.lower() == "branded"]
        .sort_values(["score", "rating"], ascending=[False, False])
        .to_dict(orient="records")
    )

    balanced = []
    # Keep strong relevance while ensuring local visibility in every small block.
    pattern = ["branded", "local", "branded", "local", "branded", "local"]
    while local_pool or branded_pool:
        moved = False
        for slot in pattern:
            if slot == "local" and local_pool:
                balanced.append(local_pool.pop(0))
                moved = True
            elif slot == "branded" and branded_pool:
                balanced.append(branded_pool.pop(0))
                moved = True
            if not local_pool and not branded_pool:
                break
        if not moved:
            break

    balanced_df = pd.DataFrame(balanced)
    balanced_df["fair_rank"] = range(1, len(balanced_df) + 1)
    return balanced_df


def build_catalog():
    print("Loading dataset...")
    base_df = pd.read_csv(DATASET_PATH)
    catalog = choose_catalog_slice(base_df)
    scored = prepare_model_features(catalog)

    print("Running model predictions...")
    preds, model_used = predictor.predict(scored)
    scored["score"] = preds
    scored = build_fair_ranking(scored)
    scored["is_low_stock"] = scored["stock_qty"] <= 5
    scored["size_list"] = scored["size_options"].fillna("").map(lambda value: [part.strip() for part in value.split("|") if part.strip()])
    scored["discounted"] = scored["discount_percent"] > 0
    scored["brand_badge"] = scored["brand_type"].str.title()
    scored["price_display"] = scored["price"].map(lambda value: f"{int(value):,}")
    scored["original_price_display"] = scored["original_price"].map(lambda value: f"{int(value):,}")
    scored["rating_display"] = scored["rating"].map(lambda value: f"{value:.1f}")
    scored["review_count_display"] = scored["review_count"].map(lambda value: f"{int(value):,}")
    scored["score_display"] = scored["score"].map(lambda value: f"{value:.3f}")
    tshirt_replacements = ["shirt-front.jpg", "shirt-5.jpg", "shirt-6.jpg", "shirt-back.jpg"]
    tshirt_mask = scored["category"].str.lower() == "tshirt"
    tshirt_indices = list(scored.index[tshirt_mask])
    for position, row_index in enumerate(tshirt_indices):
        scored.at[row_index, "image_filename"] = tshirt_replacements[position % len(tshirt_replacements)]
    print("App ready!")
    return scored.sort_values("fair_rank", ascending=True).reset_index(drop=True), model_used


def product_image_palette(category):
    palettes = {
        "shirt": ("#0f172a", "#334155", "#f8fafc"),
        "tshirt": ("#1d4ed8", "#60a5fa", "#eff6ff"),
        "jeans": ("#1f2937", "#6b7280", "#f9fafb"),
        "trousers": ("#3f3f46", "#a1a1aa", "#fafaf9"),
    }
    return palettes.get(category.lower(), ("#78350f", "#f59e0b", "#fffbeb"))


def garment_fill(color_name, category):
    swatches = {
        "black": "#1f2937",
        "navy": "#1d4ed8",
        "olive": "#4d7c0f",
        "sand": "#b08968",
        "charcoal": "#52525b",
        "white": "#f8fafc",
    }
    default_fill = {
        "shirt": "#475569",
        "tshirt": "#2563eb",
        "jeans": "#1e3a8a",
        "trousers": "#57534e",
    }
    return swatches.get((color_name or "").lower(), default_fill.get((category or "").lower(), "#64748b"))


def fabric_pattern(fabric_name):
    fabric_name = (fabric_name or "").lower()
    if "denim" in fabric_name:
        return '<path d="M0 0H640V640H0Z" fill="url(#diagPattern)" fill-opacity="0.22"/>'
    if "linen" in fabric_name or "oxford" in fabric_name:
        return '<path d="M0 0H640V640H0Z" fill="url(#dotPattern)" fill-opacity="0.18"/>'
    if "twill" in fabric_name:
        return '<path d="M0 0H640V640H0Z" fill="url(#diagPattern)" fill-opacity="0.14"/>'
    return ""


def product_shape_svg(category, garment_color, accent_color):
    category = (category or "").lower()
    if category == "shirt":
        return f"""
<g transform="translate(170 120)">
  <path d="M120 20 170 55 210 28 260 78 220 132 190 112 190 360 50 360 50 112 20 132-20 78 30 28 70 55 120 20Z"
        fill="{garment_color}"/>
  <path d="M103 20h34l10 36-27 26-27-26 10-36Z" fill="{accent_color}" fill-opacity="0.22"/>
  <path d="M86 150h68M86 190h68M86 230h68" stroke="{accent_color}" stroke-opacity="0.18" stroke-width="8" stroke-linecap="round"/>
</g>"""
    if category == "tshirt":
        return f"""
<g transform="translate(175 135)">
  <path d="M105 18 155 42 198 22 248 78 210 122 178 96 178 350 32 350 32 96 0 122-38 78 12 22 55 42 105 18Z"
        fill="{garment_color}"/>
  <path d="M87 18h36c4 18-6 34-18 43-12-9-22-25-18-43Z" fill="{accent_color}" fill-opacity="0.2"/>
</g>"""
    if category == "jeans":
        return f"""
<g transform="translate(210 118)">
  <path d="M55 20h110l22 122-40 220h-54l-16-140-16 140H7L-33 142 55 20Z"
        fill="{garment_color}"/>
  <path d="M86 20v88M24 145h142" stroke="{accent_color}" stroke-opacity="0.18" stroke-width="8" stroke-linecap="round"/>
</g>"""
    return f"""
<g transform="translate(208 118)">
  <path d="M40 20h125l18 102-18 240h-56l-14-158-14 158H25L7 122 40 20Z"
        fill="{garment_color}"/>
  <path d="M70 20v82M127 20v82M27 145h150" stroke="{accent_color}" stroke-opacity="0.18" stroke-width="8" stroke-linecap="round"/>
</g>"""


def product_to_dict(row):
    product = row.to_dict()
    product["display_image"] = url_for("catalog_image", product_id=int(row["product_id"]))
    product["share_path"] = url_for("product_detail", product_id=int(row["product_id"]))
    product["stock_class"] = "low" if product["is_low_stock"] else "normal"
    return product


def get_product(product_id):
    matches = catalog_df.loc[catalog_df["product_id"] == product_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def category_backgrounds(category):
    backgrounds = {
        "shirt": [
            ("#f3efe7", "#d8d0c4"),
            ("#eef2f6", "#cfd7e3"),
            ("#f6f1ea", "#d6c4b3"),
            ("#ece7df", "#d8dbdf"),
        ],
        "tshirt": [
            ("#f6f6f2", "#dad8d0"),
            ("#f0f4f7", "#d4dee6"),
            ("#f6efe8", "#e0cfc0"),
            ("#eff0eb", "#d6d7cf"),
        ],
        "jeans": [
            ("#eef2f6", "#c7d3df"),
            ("#f4f5f7", "#d7dbe4"),
            ("#eef0ed", "#cfd5d0"),
            ("#f7f3ee", "#d9d0c7"),
        ],
        "trousers": [
            ("#f3f1eb", "#d4d1c6"),
            ("#eff2f1", "#ced5d4"),
            ("#f4efec", "#ddd2ca"),
            ("#eef0f4", "#d0d7e1"),
        ],
    }
    return backgrounds.get((category or "").lower(), [("#f2f2f2", "#d8d8d8")])


def render_catalog_image(product):
    image_filename = str(product.get("image_filename", "")).strip()
    source_path = BASE_DIR / "static" / "product_images" / image_filename
    if not image_filename or not source_path.exists():
        return None

    base = Image.open(source_path).convert("RGBA")
    canvas_size = 900
    variant_seed = int(product["product_id"])
    gradients = category_backgrounds(product.get("category", ""))
    gradient_start, gradient_end = gradients[variant_seed % len(gradients)]

    background = Image.new("RGBA", (canvas_size, canvas_size), gradient_start)
    bg_draw = ImageDraw.Draw(background)
    for y in range(canvas_size):
        ratio = y / max(canvas_size - 1, 1)
        start_rgb = tuple(int(gradient_start[i:i + 2], 16) for i in (1, 3, 5))
        end_rgb = tuple(int(gradient_end[i:i + 2], 16) for i in (1, 3, 5))
        row_color = tuple(int(start_rgb[i] * (1 - ratio) + end_rgb[i] * ratio) for i in range(3))
        bg_draw.line((0, y, canvas_size, y), fill=row_color)

    panel = Image.new("RGBA", (canvas_size - 80, canvas_size - 80), (255, 255, 255, 72))
    panel = panel.filter(ImageFilter.GaussianBlur(radius=2))
    background.alpha_composite(panel, (40, 40))

    max_width = 620 + (variant_seed % 5) * 18
    max_height = 620 + (variant_seed % 4) * 22
    object_image = ImageOps.contain(base, (max_width, max_height))
    angle = ((variant_seed % 7) - 3) * 0.7
    object_image = object_image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

    shadow = Image.new("RGBA", object_image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (16, object_image.size[1] - 82, object_image.size[0] - 16, object_image.size[1] - 28),
        radius=30,
        fill=(0, 0, 0, 70),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))

    x_offset = 140 + (variant_seed % 4) * 28
    y_offset = 120 + (variant_seed % 3) * 18
    background.alpha_composite(shadow, (x_offset - 10, y_offset + 24))
    background.alpha_composite(object_image, (x_offset, y_offset))

    badge_draw = ImageDraw.Draw(background)
    badge_draw.rounded_rectangle((56, 56, 844, 844), radius=46, outline=(255, 255, 255, 90), width=2)

    output = BytesIO()
    background.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    output.seek(0)
    return output


def get_favorites():
    return session.setdefault("favorites", [])


def get_cart():
    return session.setdefault("cart", {})


def enrich_products_with_state(products):
    favorites = {int(product_id) for product_id in get_favorites()}
    cart = {int(product_id): qty for product_id, qty in get_cart().items()}
    for product in products:
        product["is_favorite"] = int(product["product_id"]) in favorites
        product["cart_quantity"] = cart.get(int(product["product_id"]), 0)
    return products


def get_filters():
    return {
        "query": request.args.get("q", "").strip(),
        "category": request.args.get("category", "").strip(),
        "brand_name": request.args.get("brand_name", "").strip(),
        "stock": request.args.get("stock", "").strip(),
        "sort": request.args.get("sort", "recommended").strip(),
    }


def filter_products():
    products = catalog_df.copy()
    filters = get_filters()

    if filters["query"]:
        query = filters["query"].lower()
        search_mask = (
            products["product_name"].str.lower().str.contains(query)
            | products["description"].str.lower().str.contains(query)
            | products["brand_name"].str.lower().str.contains(query)
            | products["occasion"].str.lower().str.contains(query)
        )
        products = products.loc[search_mask]

    if filters["category"]:
        products = products.loc[products["category"] == filters["category"]]

    if filters["brand_name"]:
        products = products.loc[products["brand_name"] == filters["brand_name"]]

    if filters["stock"] == "low":
        products = products.loc[products["is_low_stock"]]
    elif filters["stock"] == "available":
        products = products.loc[products["stock_qty"] > 0]

    if filters["sort"] == "price_low":
        products = products.sort_values(["price", "score"], ascending=[True, False])
    elif filters["sort"] == "price_high":
        products = products.sort_values(["price", "score"], ascending=[False, False])
    elif filters["sort"] == "rating":
        products = products.sort_values(["rating", "score"], ascending=[False, False])
    else:
        products = products.sort_values(["fair_rank", "score"], ascending=[True, False])

    return [product_to_dict(row) for _, row in products.iterrows()], filters


def catalog_summary(products):
    return {
        "local_count": sum(1 for product in products if product["brand_type"] == "local"),
        "branded_count": sum(1 for product in products if product["brand_type"] == "branded"),
        "low_stock_count": sum(1 for product in products if product["is_low_stock"]),
    }


def cart_items():
    items = []
    subtotal = 0
    savings = 0

    for product_id, qty in get_cart().items():
        product = get_product(int(product_id))
        if product is None:
            continue

        quantity = max(1, int(qty))
        line_total = int(product["price"]) * quantity
        line_savings = max(0, int(product["original_price"]) - int(product["price"])) * quantity
        subtotal += line_total
        savings += line_savings

        item = product_to_dict(product)
        item["quantity"] = quantity
        item["line_total"] = line_total
        item["line_total_display"] = f"{line_total:,}"
        items.append(item)

    shipping = 0 if subtotal >= 2500 or subtotal == 0 else 149
    total = subtotal + shipping
    return items, {
        "subtotal": subtotal,
        "subtotal_display": f"{subtotal:,}",
        "shipping": shipping,
        "shipping_display": "Free" if shipping == 0 and subtotal > 0 else f"{shipping:,}",
        "savings": savings,
        "savings_display": f"{savings:,}",
        "total": total,
        "total_display": f"{total:,}",
    }


def favorite_products():
    products = []
    for product_id in {int(item) for item in get_favorites()}:
        product = get_product(product_id)
        if product is not None:
            products.append(product_to_dict(product))
    return enrich_products_with_state(sorted(products, key=lambda item: item["fair_rank"]))


def address_preview(user):
    if not user:
        return ""
    fields = [
        user.get("door_number", ""),
        user.get("street_name", ""),
        user.get("place", ""),
        user.get("state", ""),
        user.get("country", ""),
    ]
    return ", ".join([field for field in fields if field])


@app.context_processor
def inject_globals():
    return {
        "logged_in_user": current_user(),
        "cart_count": sum(int(qty) for qty in get_cart().values()) if "cart" in session else 0,
        "favorite_count": len(get_favorites()) if "favorites" in session else 0,
    }


predictor = load_predictor()
catalog_df, model_used = build_catalog()


@app.route("/")
def landing():
    if login_required():
        return redirect(url_for("home"))
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        field_names = [
            "username",
            "password",
            "email",
            "phone",
            "door_number",
            "street_name",
            "place",
            "state",
            "country",
        ]
        form = {key: request.form.get(key, "").strip() for key in field_names}
        if any(not value for value in form.values()):
            flash("Please complete all signup fields.", "error")
            return render_template("signup.html", form=form)

        users = load_users()
        if form["username"] in users:
            flash("That username already exists. Please choose another one.", "error")
            return render_template("signup.html", form=form)

        users[form["username"]] = {
            "username": form["username"],
            "password_hash": generate_password_hash(form["password"]),
            "email": form["email"],
            "phone": form["phone"],
            "door_number": form["door_number"],
            "street_name": form["street_name"],
            "place": form["place"],
            "state": form["state"],
            "country": form["country"],
        }
        save_users(users)
        flash("Signup successful. Please log in to continue.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html", form={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = load_users().get(username)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "error")
            return render_template("login.html", form={"username": username})

        session["user"] = username
        session.setdefault("cart", {})
        session.setdefault("favorites", [])
        flash(f"Welcome back, {username}.", "success")
        return redirect(url_for("home"))

    return render_template("login.html", form={})


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("landing"))


@app.route("/home")
def home():
    if not login_required():
        return redirect(url_for("login"))

    products, filters = filter_products()
    products = enrich_products_with_state(products)
    categories = sorted(catalog_df["category"].dropna().unique())
    brands = sorted(catalog_df["brand_name"].dropna().unique())
    return render_template(
        "home.html",
        products=products,
        filters=filters,
        categories=categories,
        brands=brands,
        model=model_used,
        summary=catalog_summary(products),
    )


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    if not login_required():
        return redirect(url_for("login"))

    product = get_product(product_id)
    if product is None:
        flash("Product not found.", "error")
        return redirect(url_for("home"))

    product_dict = enrich_products_with_state([product_to_dict(product)])[0]
    return render_template(
        "product_detail.html",
        product=product_dict,
        address_preview=address_preview(current_user()),
    )


@app.route("/favorites")
def favorites():
    if not login_required():
        return redirect(url_for("login"))
    return render_template("favorites.html", products=favorite_products())


@app.route("/favorites/toggle/<int:product_id>", methods=["POST"])
def toggle_favorite(product_id):
    if not login_required():
        return redirect(url_for("login"))

    favorites_list = get_favorites()
    if product_id in [int(item) for item in favorites_list]:
        session["favorites"] = [item for item in favorites_list if int(item) != product_id]
        flash("Removed from favourites.", "success")
    else:
        favorites_list.append(str(product_id))
        session["favorites"] = favorites_list
        flash("Saved to favourites.", "success")

    return redirect(request.referrer or url_for("home"))


@app.route("/cart")
def cart():
    if not login_required():
        return redirect(url_for("login"))
    items, totals = cart_items()
    return render_template("cart.html", items=enrich_products_with_state(items), totals=totals)


@app.route("/cart/add/<int:product_id>", methods=["POST"])
def add_to_cart(product_id):
    if not login_required():
        return redirect(url_for("login"))

    product = get_product(product_id)
    if product is None:
        flash("Product not found.", "error")
        return redirect(url_for("home"))

    quantity = max(1, int(request.form.get("quantity", 1)))
    cart_map = get_cart()
    product_key = str(product_id)
    cart_map[product_key] = min(int(product["stock_qty"]), cart_map.get(product_key, 0) + quantity)
    session["cart"] = cart_map
    flash("Added to cart.", "success")
    return redirect(request.referrer or url_for("home"))


@app.route("/cart/update/<int:product_id>", methods=["POST"])
def update_cart(product_id):
    if not login_required():
        return redirect(url_for("login"))

    cart_map = get_cart()
    quantity = int(request.form.get("quantity", 1))
    if quantity <= 0:
        cart_map.pop(str(product_id), None)
    else:
        product = get_product(product_id)
        if product is not None:
            cart_map[str(product_id)] = min(quantity, int(product["stock_qty"]))
    session["cart"] = cart_map
    flash("Cart updated.", "success")
    return redirect(url_for("cart"))


@app.route("/cart/remove/<int:product_id>", methods=["POST"])
def remove_from_cart(product_id):
    if not login_required():
        return redirect(url_for("login"))

    cart_map = get_cart()
    cart_map.pop(str(product_id), None)
    session["cart"] = cart_map
    flash("Removed item from cart.", "success")
    return redirect(url_for("cart"))


@app.route("/checkout", methods=["GET", "POST"])
def checkout():
    if not login_required():
        return redirect(url_for("login"))

    items, totals = cart_items()
    if not items:
        flash("Your cart is empty. Add products before checkout.", "error")
        return redirect(url_for("home"))

    if request.method == "POST":
        payment_method = request.form.get("payment_method", "").strip()
        if payment_method not in {"GPay", "Debit Card", "Cash on Delivery"}:
            flash("Choose a valid payment mode.", "error")
        else:
            session["last_order"] = {
                "items": len(items),
                "total": totals["total_display"],
                "payment_method": payment_method,
            }
            session["cart"] = {}
            flash(f"Order placed successfully using {payment_method}.", "success")
            return redirect(url_for("home"))

    return render_template(
        "checkout.html",
        items=enrich_products_with_state(items),
        totals=totals,
        address_preview=address_preview(current_user()),
    )


@app.route("/product-image/<int:product_id>.svg")
def product_image(product_id):
    product = get_product(product_id)
    if product is None:
        title = "MR Product"
        subtitle = "MR"
        palette = ("#111827", "#374151", "#f9fafb")
    else:
        title = product["brand_name"]
        subtitle = product["category"].title()
        palette = product_image_palette(product["category"])

    start, end, text = palette
    category = product["category"] if product is not None else "shirt"
    color_name = product["color"] if product is not None else "black"
    fabric_name = product["fabric"] if product is not None else ""
    fit_name = product["fit"] if product is not None else ""
    garment_color = garment_fill(color_name, category)
    garment_svg = product_shape_svg(category, garment_color, text)
    pattern_overlay = fabric_pattern(fabric_name)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0%" stop-color="{start}"/>
<stop offset="100%" stop-color="{end}"/>
</linearGradient>
<linearGradient id="floor" x1="0" y1="0" x2="1" y2="0">
<stop offset="0%" stop-color="rgba(255,255,255,0.00)"/>
<stop offset="50%" stop-color="rgba(255,255,255,0.16)"/>
<stop offset="100%" stop-color="rgba(255,255,255,0.00)"/>
</linearGradient>
<pattern id="diagPattern" width="24" height="24" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<line x1="0" y1="0" x2="0" y2="24" stroke="#ffffff" stroke-opacity="1" stroke-width="6"/>
</pattern>
<pattern id="dotPattern" width="18" height="18" patternUnits="userSpaceOnUse">
<circle cx="5" cy="5" r="2" fill="#ffffff" fill-opacity="1"/>
</pattern>
</defs>
<rect width="640" height="640" rx="42" fill="url(#bg)"/>
<rect x="58" y="58" width="524" height="524" rx="34" fill="none" stroke="rgba(255,255,255,0.14)" stroke-width="2"/>
<ellipse cx="320" cy="468" rx="165" ry="38" fill="rgba(0,0,0,0.16)"/>
<rect x="142" y="436" width="356" height="10" rx="5" fill="url(#floor)"/>
{garment_svg}
{pattern_overlay}
<text x="88" y="110" font-family="Georgia, serif" font-size="52" font-weight="700" fill="{text}">MR</text>
<text x="88" y="470" font-family="Georgia, serif" font-size="54" font-weight="700" fill="{text}">{escape(str(title)[:18] or 'MR')}</text>
<text x="88" y="522" font-family="Segoe UI, Arial, sans-serif" font-size="28" fill="{text}" fill-opacity="0.82">{escape(str(subtitle)[:18] or 'MR')}</text>
<text x="88" y="566" font-family="Segoe UI, Arial, sans-serif" font-size="24" fill="{text}" fill-opacity="0.62">{escape(str(color_name).title())} | {escape(str(fit_name))}</text>
    </svg>"""
    return Response(svg, mimetype="image/svg+xml")


@app.route("/catalog-image/<int:product_id>.jpg")
def catalog_image(product_id):
    product = get_product(product_id)
    if product is None:
        return redirect(url_for("product_image", product_id=product_id))

    rendered = render_catalog_image(product)
    if rendered is None:
        return redirect(url_for("product_image", product_id=product_id))

    return send_file(rendered, mimetype="image/jpeg", max_age=3600)


@app.route("/share/<int:product_id>")
def share_product(product_id):
    product = get_product(product_id)
    if product is None:
        return redirect(url_for("home"))
    query = urlencode({"shared": product["product_name"]})
    return redirect(f"{url_for('product_detail', product_id=product_id)}?{query}")


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=debug_mode)
