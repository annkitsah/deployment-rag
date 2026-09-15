# Deploying the backend (Oracle Cloud Always Free + Docker)

This deploys the FastAPI backend, Ollama, and Caddy (automatic HTTPS)
together on one Oracle Cloud "Always Free" ARM VM, using Docker
Compose. This is a genuinely free, permanent setup -- not a trial --
sized to comfortably run `llama3.2:3b` alongside the API.

## Prerequisites

- An Oracle Cloud account (oracle.com/cloud/free) -- a credit card is
  required for identity verification, but you are not charged as long
  as you stay within the Always Free limits.
- A domain name (or a subdomain of one you own) you can point at the
  VM's public IP. This is required for automatic HTTPS -- Caddy cannot
  get a certificate for a bare IP address.
- An SSH key pair for connecting to the VM.

## 1. Create the VM

In the Oracle Cloud console: Compute -> Instances -> Create Instance.

- Shape: change shape, select "Ampere" -> `VM.Standard.A1.Flex`.
  Confirm the "Always Free" badge is shown.
- Give it the full free allocation available to your account (check
  the current limit in the console -- it has changed over time).
- Image: Ubuntu (22.04 or later).
- Add your SSH public key.
- Networking: use the default VCN with a public subnet, or create one.

Once running, note the VM's public IP and point your domain's DNS A
record at it.

**Firewall gotcha (very common):** Oracle Cloud blocks inbound traffic
at *two* independent layers -- the VCN's Security List/Network Security
Group (in the console) *and* the VM's own firewall (`iptables`/`ufw`).
You need to open ports 80 and 443 in **both** places, or Caddy's
Let's Encrypt certificate request will silently fail.

In the console: your VCN -> Security Lists -> add ingress rules for
TCP 80 and TCP 443 from `0.0.0.0/0`.

On the VM itself:
```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save   # if installed; otherwise persist however your image expects
```

## 2. Install Docker

SSH into the VM, then:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

(Docker Compose v2 is bundled with modern Docker as `docker compose`.)

## 3. Get the code onto the VM

```bash
git clone https://github.com/<your-username>/agentic-true-vectorless-rag.git
cd agentic-true-vectorless-rag
```

## 4. Configure production environment

```bash
cp .env.production.example .env.production
nano .env.production
```

At minimum, set:
- `MISTRAL_API_KEY` -- your real key (still needed for OCR)
- `CORS_ALLOWED_ORIGINS` -- your Vercel frontend's URL, e.g.
  `https://your-app.vercel.app`

Then edit `docker/Caddyfile` and replace `your-backend-domain.example.com`
with your real domain.

## 5. Start everything

```bash
cd docker
docker compose up -d --build
```

This builds the API image, pulls the official `ollama` and `caddy`
images, and starts all three. Caddy will automatically request a TLS
certificate for the domain in the Caddyfile on first request.

## 6. Pull the model

The Ollama container starts with no models downloaded. Pull the one
your `.env.production` references:

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

This only needs to be done once -- the model is stored in the
`ollama_models` named volume and survives container restarts.

## 7. Verify

```bash
curl https://your-backend-domain.example.com/health
```

Should return a JSON health status with `indexed_pages`. If this
fails, check `docker compose logs caddy` (certificate issues) and
`docker compose logs api` (application errors) before anything else.

## 8. Point the frontend at it

In your Next.js project on Vercel, set an environment variable (e.g.
`NEXT_PUBLIC_API_URL=https://your-backend-domain.example.com`) so the
frontend knows where to send requests.

## Updating after future changes

```bash
cd agentic-true-vectorless-rag
git pull
cd docker
docker compose up -d --build
```

Persistent data (SQLite, page files, the index snapshot, and the
downloaded Ollama model) lives in named Docker volumes and is
untouched by rebuilds -- only `docker compose down -v` would delete it.
