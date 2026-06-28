-- Tiny self-contained analytics schema for the text→SQL agent eval.
-- Committed + seeded (no secrets) so CI and local runs build an identical DB.
-- Domain: a small e-commerce store (customers, products, orders, order_items).

CREATE TABLE customers (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    country    TEXT NOT NULL,
    signup_date TEXT NOT NULL          -- ISO date
);

CREATE TABLE products (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    category  TEXT NOT NULL,
    price     REAL NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL,          -- ISO date
    status      TEXT NOT NULL           -- 'paid' | 'refunded' | 'pending'
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL
);

INSERT INTO customers (id, name, country, signup_date) VALUES
    (1, 'Alice',  'US', '2025-01-05'),
    (2, 'Bjorn',  'SE', '2025-02-11'),
    (3, 'Chitra', 'IN', '2025-02-20'),
    (4, 'Diego',  'US', '2025-03-02'),
    (5, 'Emi',    'JP', '2025-03-18');

INSERT INTO products (id, name, category, price) VALUES
    (1, 'Cinestill 800T',  'film',  18.99),
    (2, 'Kodak Portra 400','film',  15.49),
    (3, '50mm f/1.8 lens', 'lens',  125.00),
    (4, '85mm f/1.4 lens', 'lens',  399.00),
    (5, 'Tripod',          'gear',  89.00);

INSERT INTO orders (id, customer_id, order_date, status) VALUES
    (1, 1, '2025-03-01', 'paid'),
    (2, 1, '2025-03-15', 'paid'),
    (3, 2, '2025-03-20', 'refunded'),
    (4, 3, '2025-04-02', 'paid'),
    (5, 4, '2025-04-10', 'pending'),
    (6, 5, '2025-04-12', 'paid'),
    (7, 1, '2025-04-20', 'paid');

INSERT INTO order_items (id, order_id, product_id, quantity) VALUES
    (1, 1, 1, 3),
    (2, 1, 2, 2),
    (3, 2, 3, 1),
    (4, 3, 4, 1),
    (5, 4, 1, 5),
    (6, 4, 5, 1),
    (7, 5, 2, 4),
    (8, 6, 3, 2),
    (9, 7, 4, 1),
    (10, 7, 5, 2);
