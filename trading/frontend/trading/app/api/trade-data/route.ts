import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  try {
    const dataDir = path.join(process.cwd(), 'data');
    const files = fs.readdirSync(dataDir);
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    const tradeData = jsonFiles.map(file => {
      const filePath = path.join(dataDir, file);
      const fileContent = fs.readFileSync(filePath, 'utf-8');
      return JSON.parse(fileContent);
    });

    return NextResponse.json(tradeData);
  } catch (error) {
    console.error('Error reading trade data:', error);
    return NextResponse.json({ error: 'Failed to load trade data' }, { status: 500 });
  }
} 