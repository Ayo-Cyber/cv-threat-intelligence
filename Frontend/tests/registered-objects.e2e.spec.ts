import { expect, test } from "@playwright/test";

for (const width of [1280, 390]) {
  test(`registered object region at ${width}px`, async ({page}) => {
    await page.setViewportSize({width,height:600});
    await page.goto("/tests/fixtures/registered-objects.html");
    await page.getByRole("button",{name:"Open live view"}).click();
    await expect(page.getByText("LIVE PREVIEW",{exact:true})).toBeVisible();
    const region = page.getByLabel("Object region");
    const bounds = (await region.boundingBox())!;
    expect(bounds.width/bounds.height).toBeCloseTo(640/360,1);
    await page.mouse.move(bounds.x+bounds.width*.25,bounds.y+bounds.height*.25);
    await page.mouse.down();
    await page.mouse.move(bounds.x+bounds.width*.625,bounds.y+bounds.height*.75);
    await page.mouse.up();
    await page.getByLabel("Object name").fill("Laptop");
    await page.getByLabel("Confirmation time (seconds)").fill("8");
    await page.getByRole("button",{name:"Register object",exact:true}).click();
    const call = await page.evaluate(() => (window as any).calls.find((c:any)=>c.method==="register_object_region"));
    expect(call.args[2]).toEqual([160,90,400,270]);
    expect(call.args[3]).toEqual([360,640]);
    await expect(page.getByText("capturing reference",{exact:true})).toBeVisible();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
    await page.screenshot({path:`test-results/registered-objects-${width}.png`,fullPage:true});
    page.on("dialog",d=>d.accept());
    await page.getByRole("button",{name:"Remove Laptop"}).click();
    await expect(page.getByRole("button",{name:"Remove Laptop"})).toHaveCount(0);
  });
}
