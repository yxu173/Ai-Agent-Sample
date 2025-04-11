/** @type {import('prettier').Config} */
module.exports = {
  singleQuote: true,
  semi: false,
  trailingComma: 'none',
  plugins: ['prettier-plugin-tailwindcss'],
  filepath: './src/**/*.{js,ts,jsx,tsx}'
}
