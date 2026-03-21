# plato-ai-skill

## Configuração da Chave de Acesso (Access Key Setup)

Para utilizar este projeto, é necessário configurar a chave de acesso à API.

### Passo a passo

1. Copie o arquivo de exemplo de variáveis de ambiente:
   ```bash
   cp .env.example .env
   ```

2. Abra o arquivo `.env` e preencha o valor da sua chave de API:
   ```
   PLATO_API_KEY=sua_chave_aqui
   ```

3. **Nunca** faça commit do arquivo `.env` — ele já está incluído no `.gitignore` para proteger suas credenciais.

---

*For English: copy `.env.example` to `.env`, fill in `PLATO_API_KEY` with your API key, and keep the `.env` file out of version control.*