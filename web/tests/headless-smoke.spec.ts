import { test, expect } from '@playwright/test'

test('quick headless smoke without video', async ({ page }) => {
  await page.goto('/', { waitUntil: 'domcontentloaded' })
  await expect(page.getByText('Edit workspace')).toBeVisible()
  await page.getByText('Assets').click()
  await expect(page.getByText('Asset bin')).toBeVisible()
  await page.getByText('Export').click()
  await page.getByRole('button', { name: 'Render preset' }).click()
  await page.waitForTimeout(200)
  await expect(page.getByText(/Status/)).toBeVisible()
})
