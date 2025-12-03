import { test, expect, type Page } from '@playwright/test'
import path from 'path'

const timelineTrackLane = '.track-lane'

const sampleAudioPath = path.join(process.cwd(), 'public', 'samples', 'free-tone-10s.wav')
const sampleImagePath = path.join(process.cwd(), 'public', 'samples', 'mars-1280.jpg')

const waitForRenderIdle = async (page: Page) => {
  await page.waitForTimeout(400)
}

test('timeline basics, snapping, loop handles, asset drop, export', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('Edit workspace')).toBeVisible()

  // landing view
  await page.screenshot({ path: 'screenshots/edit-overview-v2.png', fullPage: true })

  // marker click moves playhead
  const firstMarker = page.locator('.marker-list li').first()
  await firstMarker.scrollIntoViewIfNeeded()
  await firstMarker.click()
  const playhead = page.locator('.playhead')
  await expect(playhead).toBeVisible()

  // drag a clip to trigger snap ghost
  const clip = page.locator('.clip').first()
  const box = await clip.boundingBox()
  if (box) {
    await page.mouse.move(box.x + 10, box.y + box.height / 2)
    await page.mouse.down()
    await page.mouse.move(box.x + 40, box.y + box.height / 2, { steps: 4 })
    await page.mouse.up()
  }

  // loop handles respond
  const startHandle = page.locator('.loop-handle.start')
  await expect(startHandle).toBeVisible()
  await startHandle.click({ force: true })

  // asset upload + drag to track
  await page.getByRole('button', { name: 'Assets' }).click()
  const input = page.locator('input[type="file"]')
  await input.setInputFiles([sampleAudioPath, sampleImagePath])
  await waitForRenderIdle(page)
  await page.screenshot({ path: 'screenshots/assets-v2.png', fullPage: true })
  const assetRow = page.locator('.asset-row').first()
  await expect(assetRow).toBeVisible()
  const sendButton = assetRow.getByRole('button', { name: /Send to/ }).first()
  await sendButton.click()
  await page.locator('.tabs .tab', { hasText: 'Edit' }).click()
  await waitForRenderIdle(page)
  const clipCount = await page.locator('.clip').count()
  expect(clipCount).toBeGreaterThan(5)

  // beat detection module
  const beatPanel = page.locator('.beat-panel')
  await expect(page.getByText('Beat detection')).toBeVisible()
  await beatPanel.scrollIntoViewIfNeeded()
  await beatPanel.getByRole('button', { name: 'Detect beats' }).click()
  await waitForRenderIdle(page)
  await page.screenshot({ path: 'screenshots/beat-detect.png', fullPage: true })

  // export preset mock
  await page.locator('.tabs .tab', { hasText: 'Export' }).click()
  await page.locator('select.ghost').selectOption('mp4')
  const renderBtn = page.getByRole('button', { name: 'Render preset' })
  await renderBtn.click()
  await page.waitForTimeout(600)

  await page.screenshot({ path: 'screenshots/export-v2.png', fullPage: true })
  await page.screenshot({ path: 'screenshots/timeline-v2.png', fullPage: true })
})
